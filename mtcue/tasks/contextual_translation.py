from dataclasses import dataclass, field

import logging
import numpy as np
import os
import json
import glob
from functools import partial

from fairseq import search, utils

from fairseq.data import (
    AppendTokenDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)
from mtcue.data import MTCueDataset

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask

from torch.utils.data import TensorDataset
from torch import FloatStorage, FloatTensor, randn, IntTensor, IntStorage
import torch

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


def load_mtcue_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        context_model,
        no_doc_context,
        doc_context_len,
        no_meta_context,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        prepend_bos_src=None,
        list_of_ablations=None,
):
    """Load dataset.

    Args:
        src, tgt: lang codes
        """

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    # infer langcode
    if split_exists(split, src, tgt, src, data_path):
        prefix = os.path.join(data_path, "{}.{}-{}.".format(split, src, tgt))
    elif split_exists(split, tgt, src, src, data_path):
        prefix = os.path.join(data_path, "{}.{}-{}.".format(split, tgt, src))
    else:
        raise FileNotFoundError(
            "Dataset not found: {} ({})".format(split, data_path)
        )

    src_dataset = data_utils.load_indexed_dataset(
        prefix + src, src_dict, dataset_impl
    )
    if truncate_source:
        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(src_dataset, src_dict.eos()),
                max_source_positions - 1,
            ),
            src_dict.eos(),
        )

    tgt_dataset = data_utils.load_indexed_dataset(
        prefix + tgt, tgt_dict, dataset_impl
    )
    cxt_model = f"{prefix}{context_model}"
    logging.info(f"--- Loading context from {cxt_model}")
    """
    Context is saved in up to four files total:
    {cxt_model}.json: contains the context model dims and indices for metadata (if any)
    {cxt_model}.doc.bin: document-level context
    {cxt_model}.meta.bin: metadata context
    
    """
    with open(f'{cxt_model}.json') as f:
        cmap = json.load(f)

    def load_vec_storage(filename, uniq_len):
        return FloatTensor(
            FloatStorage.from_file(
                filename, shared=True, size=uniq_len * cmap['embed_dim'])).reshape(uniq_len, cmap['embed_dim'])

    def load_idx_storage(filename, width):
        return IntTensor(
            IntStorage.from_file(
                filename, shared=True, size=cmap["dataset_len"] * width)) \
            .reshape(cmap["dataset_len"], width).type(torch.long)
    if not no_doc_context:
        document_vectors = load_vec_storage(f"{cxt_model}.{src}.bin", uniq_len=cmap[f"{src}_len_uniq"])
        document_idx = load_idx_storage(f"{cxt_model}.{src}.idx", width=cmap[f"{src}_len"])

    if not no_meta_context:
        metadata_vectors = load_vec_storage(f"{cxt_model}.cxt.bin", uniq_len=cmap[f"cxt_len_uniq"])
        metadata_idx = load_idx_storage(f"{cxt_model}.cxt.idx", width=cmap[f"cxt_len"])

    cxt = {
        'doc': TensorDataset(document_vectors) if not no_doc_context else None,
        'doc_idx': document_idx if not no_doc_context else None,
        'meta': TensorDataset(metadata_vectors) if not no_meta_context else None,
        'meta_idx': metadata_idx if not no_meta_context else None,
    }
    logging.info(f"--- Context loaded: {cxt}")

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )
    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return MTCueDataset(
        cxt,
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        context_mapper=cmap,
        list_of_ablations=list_of_ablations

    )


@dataclass
class MTCueConfig(TranslationConfig):
    list_of_ablations: str = field(
        default="", metadata={"help": "list of ablations to run, separated by comma"}
    )
    context_model: str = field(
        default='sentence-transformers',
        metadata={"choices": ['distilbert', 'multilingual', 'sentence-transformers'],
                  "help": 'Which context embeddings should be used'}
    )
    no_doc_context: bool = field(
        default=False, metadata={"help": 'use document context'}
    )
    doc_context_len: int = field(
        default=6, metadata={"help": 'length of document context'}
    )
    no_meta_context: bool = field(
        default=False, metadata={"help": 'use metadata context'}
    )

@register_task("contextual_translation", dataclass=MTCueConfig)
class ContextualTranslationTask(TranslationTask):
    cfg: MTCueConfig

    @classmethod
    def setup_task(cls, cfg: MTCueConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Overwriting since cfg is different

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def build_generator(
            self,
            models,
            args,
            seq_gen_cls=None,
            extra_gen_cls_kwargs=None,
            prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from mtcue.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
                sum(
                    int(cond)
                    for cond in [
                        sampling,
                        diverse_beam_groups > 0,
                        match_source_len,
                        diversity_rate > 0,
                    ]
                )
                > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Overwriting to use a different function for loading dataset

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_mtcue_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            no_doc_context=self.cfg.no_doc_context,
            doc_context_len=self.cfg.doc_context_len,
            no_meta_context=self.cfg.no_meta_context,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            context_model=self.cfg.context_model,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            list_of_ablations=self.cfg.list_of_ablations,
        )

    def build_dataset_for_inference(self, cxt_vectors, src_tokens, src_lengths, constraints=None):
        """Builds dataset without targets. Overwriting due to inclusion of cxt_vectors"""
        return MTCueDataset(
            cxt_vectors,
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
            list_of_ablations=self.cfg.list_of_ablations
        )
