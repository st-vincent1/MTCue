
# cxt2vec

## About

This repository contains the code to vectorise contexts to use in models such as MTCue and LMCue.

## Getting Started


### Prerequisites

To run the software, you need to install the following packages (preferably in an Anaconda environment or similar):
* sentence_transformers: `pip install sentence-transformers`

### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/st-vincent1/cxt2vec.git
   ```
2. Navigate to the repository
   ```sh
   cd cxt2vec
   ```
3. Run the embedding code:
   ```sh
   python main.py --path [path to data] --dest_path [path to output embeddings] --suffixes [comma-separated suffixes of data to embed] --model [context model]
   ```

## Usage

For a data folder which looks like this:
```
data
 |- en-pl
     |- train.en
     |- train.pl
     |- valid.en
     |- valid.pl
     |- test.en
     |- test.pl
     |- context
         |- valid.speaker_gender.cxt
         |- valid.formality.cxt
         |- valid.writer_names.cxt
         |- ...
         |- valid.0.en
         |- valid.1.en
         |- ...
         |- test.0.en
         |- ...
         |- train.speaker_gender.cxt
         |- train.plot.cxt
         |- ...
 |- en-de
 |- ...
```

The following code
```sh
python main.py --paths data/en-pl --dest_path data/en-pl/embeddings --suffixes cxt,pl --model minilm
```
Finds all contexts at `data/en-pl/context` which match the given suffixes in order (`cxt`, then `pl`), embeds *all* found contexts into a single file per split (train/valid/test) per suffix. A `.json` file is also produced which contains metadata necessary to later interpret the files in training code.

The result will therefore look like so:

```
data
 |- en-pl
     |- embeddings
         |- train.minilm.cxt.bin
         |- train.minilm.cxt.idx
         |- train.minilm.en.bin
         |- train.minilm.en.idx
         |- train.minilm.json
         |- valid.minilm.cxt.bin
         |- valid.minilm.cxt.idx
         |- valid.minilm.en.bin
         |- valid.minilm.en.idx
         |- valid.minilm.json
         |- test.minilm.cxt.bin
         |- test.minilm.cxt.idx
         |- test.minilm.en.bin
         |- test.minilm.en.idx
         |- test.minilm.json
```

### Training models on this data
See examples in e.g. [github.com/st-vincent1/MTCue](MTCue repository) for how to use the embeddings to train models.


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
