# Author: Sebastian Vincent
# Date: 3 Feb 2023
# License: MIT (see repository for details)
#
# When run, program embeds contexts at given filepath and with given suffixes into TorchStorage files


import torch
import json
import transformers as ppb
import glob
import os
from tqdm import tqdm
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

BSZ = 16


# Mean Pooling - Take attention mask into account for correct averaging
# Taken from https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
def mean_pooling(model_output, attention_mask):
    # Sums feature vectors, divides embeddings by that sum
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ContextEmbedding:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        self.model_name = model_name

        match model_name:
            case "distilbert":
                self.model = ppb.DistilBertModel.from_pretrained('distilbert-base-cased').to(self.device)
                self.tokenizer = ppb.DistilBertTokenizer.from_pretrained('distilbert-base-cased')
                self.mean_pooling = False
                self.d = 768
            case "distilbert-multilingual":
                self.model = ppb.DistilBertModel.from_pretrained('distilbert-base-multilingual-cased').to(self.device)
                self.tokenizer = ppb.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
                self.mean_pooling = False
                self.d = 768
            case "mpnet":
                self.model = ppb.AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(self.device)
                self.tokenizer = ppb.AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
                self.mean_pooling = True
                self.d = 768
            case "mpnet-multilingual":
                self.model = ppb.AutoModel.from_pretrained(
                    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2').to(self.device)
                self.tokenizer = ppb.AutoTokenizer.from_pretrained(
                    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
                self.mean_pooling = True
                self.d = 768
            case "minilm":
                self.model = ppb.AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2').to(self.device)
                self.tokenizer = ppb.AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
                self.mean_pooling = True
                self.d = 384
            case "minilm-multilingual":
                self.model = ppb.AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2').to(self.device)
                self.tokenizer = ppb.AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
                self.mean_pooling = True
                self.d = 384
            case "hash":
                self.hashmap = {}
                self.d = 384

    def hash(self, sentence):
        """ Hashes a sentence to a unique torch tensor.
        WARNING: This function will only map contexts to the same hashes within one run.
        It is an error to embed training/validation and test data separately with hash. All three should be embedded
        at the same time. """
        try:
            return self.hashmap[sentence]
        except KeyError:
            self.hashmap[sentence] = torch.randn([self.d]).to(self.device)
            return self.hashmap[sentence]

    def embed_contexts(self, context_dir, out_filename, idx_filename, json_metadata, tag):
        logging.info(f"--- Loading in all sentences to create a unique set...")
        all_sentences = []
        for file_idx, filepath in enumerate(sorted(context_dir)):
            with open(filepath) as f:
                sentences = f.readlines()
                sentences = [s[:-1] for s in sentences]
            all_sentences += sentences
        unique_sentences = list(set(all_sentences))
        json_metadata = json_metadata | {f'{tag}_len_uniq': len(unique_sentences)}
        logging.info(f"--- Found {len(all_sentences)} sentences, {len(unique_sentences)} unique sentences")

        binary_buffer = torch.FloatTensor(
            torch.FloatStorage.from_file(out_filename, shared=True, size=len(unique_sentences) * self.d)) \
            .reshape(len(unique_sentences), self.d).fill_(0)

        logging.info(f"--- Embedding unique sentences...")
        sent2embid = {}
        for i in tqdm(range(0, len(unique_sentences), BSZ)):
            if "hash" in self.model_name:
                embeddings = torch.stack([self.hash(s) for s in unique_sentences[i:i + BSZ]])
            else:
                batch = self.tokenizer(unique_sentences[i:i + BSZ],
                                       add_special_tokens=True,
                                       padding=True,
                                       truncation=True,
                                       return_tensors='pt').to(self.device)
                with torch.no_grad():
                    if self.mean_pooling:
                        embeddings = self.model(**batch)
                        embeddings = mean_pooling(embeddings, batch['attention_mask'])
                    else:
                        embeddings = self.model(batch['input_ids'], attention_mask=batch['attention_mask'])[0][:, 0, :]
            binary_buffer[i:i + BSZ] = embeddings
            sent2embid.update({s: i + j for j, s in enumerate(unique_sentences[i:i + BSZ])})
        try:
            binary_buffer[sent2embid[""]] = torch.zeros(self.d)  # this will later get trigger for padding
            json_metadata[f"{tag}_pad_idx"] = sent2embid[""]
        except KeyError:
            logging.warning("No pad idx selected for this dataset. Assigning pad_idx to first available number.")
            json_metadata[f"{tag}_pad_idx"] = len(unique_sentences)
            pass

        logging.info(f"--- Creating a binary file with the embeddings and a mapping of sentence to ID...")
        idx_buffer = torch.IntStorage.from_file(idx_filename, shared=True, size=len(sentences) * len(context_dir))
        idx_buffer = torch.IntTensor(idx_buffer).reshape(len(sentences), len(context_dir))

        for file_idx, filepath in enumerate(sorted(context_dir)):
            cxt_variable = os.path.basename(filepath).split('.')[1]
            logging.info(f"Embedding {cxt_variable}")
            json_metadata[cxt_variable] = file_idx
            # Read sentences from one context file
            with open(filepath) as f:
                sentences = f.readlines()
                sentences = [s[:-1] for s in sentences]
            # Embed sentences
            for i in tqdm(range(0, len(sentences), BSZ)):
                batch = sentences[i:i + BSZ]
                idx_buffer[i:i + BSZ, file_idx] = torch.tensor([sent2embid[s] for s in batch])
        return json_metadata | {'dataset_len': len(sentences)}

    def embeddings_to_float_storage(self, input_dir, output_dir, prefix, suffixes):
        """Produces embeddings for contexts at {input_dir}/context/{prefix}*
        and saves them directly into a FloatStorage tensor at {output_dir}/{prefix}.cxt.{bin,idx}
        Saves a .json file with num_samples and num_contexts to {output_dir}/{prefix}.json"""

        def embed_(list_of_paths, tag, json_metadata):
            if not list_of_paths:
                logging.warning(f"--- No context files found for {tag}. skipping...")
                return json_metadata
            else:
                json_metadata = json_metadata | {f'{tag}_len': len(list_of_paths)}
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                out_filename = os.path.join(output_dir, f"{prefix}.{self.model_name}.{tag}.bin")
                idx_filename = os.path.join(output_dir, f"{prefix}.{self.model_name}.{tag}.idx")
                if os.path.exists(out_filename):
                    logging.warning(f"--- Binarised file for {prefix} already exists. skipping...")
                    raise FileNotFoundError
                logging.info(f"--- Scrapping data from {list_of_paths} and saving to {out_filename}...")

                json_metadata = self.embed_contexts(list_of_paths, out_filename, idx_filename, json_metadata, tag)
                return json_metadata

        # 1. Embed documents
        json_metadata = {'embed_dim': self.d}
        suffixes = suffixes.split(',')
        if any(l in suffixes for l in ['de', 'ru', 'fr', 'pl']) and "multilingual" not in self.model_name: # Weak capture as more languages exist
            logging.warning("--- WARNING: using non-multilingual model to embed non-English language text")
        
        for suffix in suffixes: 
            _dir = glob.glob(os.path.join(input_dir, "context", f"{prefix}.*.{suffix}"))

            json_metadata = embed_(_dir, suffix, json_metadata)


        # Save metadata to json
        with open(os.path.join(output_dir, f"{prefix}.{self.model_name}.json"), 'w+') as f:
            json.dump(json_metadata, f)


if __name__ == '__main__':
    """Produces embeddings for use as context input in MTCue etc.
    args.path is a list of paths to seek contexts from. """
    parser = ArgumentParser()
    parser.add_argument("--path",
                        required=True,
                        help="Path to data. assumes data at path has folder context/")
    parser.add_argument("--dest_path",
                        help="Path to save new tensor in.")
    parser.add_argument("--model",
                        required=True,
                        help="Model to use for embedding.")
    parser.add_argument("--suffixes",
                        default="en,cxt", help="suffixes to embed, separated by comma.")
    args = parser.parse_args()
    logging.info(f"Embedding with {args.model}")
    x = ContextEmbedding(args.model)

    for prefix in ['dev', 'valid', 'test', 'train', 'test_unseen']: # prefixes for context files
        try:
            x.embeddings_to_float_storage(path, args.dest_path, prefix=prefix, suffixes=args.suffixes)
        except FileNotFoundError:
            logging.warning(f"Not found {path} (or already done). Skipping")
            pass
