# Usage Guide for MTCue

This guide explains how to use the code in this repository. Before using the code, please make sure you have installed the necessary dependencies as described in the [INSTALL.md](INSTALL.md) file.

## Training

The following instructions are described for any [src]-[tgt] pair and include optional instructions if one wishes to train models in both directions using the same dataset.


### Data preparation

The following instructions are meant to explain using MTCue on new data. To use the data from the paper, follow the instructions in [INSTALL.md](INSTALL.md).

1. Put your desired dataset in the `data/` folder with the following hierarchy, where `src`, `tgt` are the source and target language respectively:
- Source/target files in `data/[src]-[tgt]/{train,dev,test}.{src,tgt}`
- Context files in `data/[src]-[tgt]/context/{train,dev,test}.[context_name].{cxt,[src],[tgt]}`, where `cxt` indicates an "other" metadata type, [src] is source context and [tgt] is target context. Source context is used when training forward models and target context is used when training backward ([tgt] -> [src]) models. None of the contexts is required per se, but this needs specifying when making embeddings/training.
2. To create embeddings for the forward model, run `bash cxt2vec/main.py --path data/[src]-[tgt] --dest_path data/[src]-[tgt]/embeddings_[src] --suffixes [src],cxt --model minilm`
3. To create embeddings for the backward model, run `bash cxt2vec/main.py --path data/[src]-[tgt] --dest_path data/[src]-[tgt]/embeddings_[tgt] --suffixes [tgt],cxt --model minilm`
4. Once embeddings are created, run `bash scripts/mtcue/prepare.sh --src [src] --tgt [tgt]` for the forward model (swap [src] and [tgt] for backward model)

### Training
To train, run `bash scripts/mtcue/finetune.sh --src [src] --tgt [tgt] --arch [mtcue/mtcue-big]`.
Note that the repository offers the possiblity to train an **MTCue-big** model which is ~4x bigger and offers slightly better translation performance (+1-2 BLEU) and similar performance on the contextual tasks. Hyperparameters are pre-selected within the file but can be modified.

### Evaluation 
To evaluate, run `bash scripts/mtcue/evaluate.sh --src [src] --tgt [tgt]`.


### Inference

To use the model for inference on new data:
1. Put your data in the `data/evaluation/my_data` repository, with the same hierarchy as training data:
- `data/evaluation/my_data/test.{[src],[tgt]}` for source/target files and `data/evaluation/my_data/context/test.[context_name].{cxt,[src],[tgt]}` for context files.
2. Run `bash scripts/mtcue/inference_prepare.sh --src [src] --tgt [tgt] --path data/evaluation/[my_data]`. The binary files will be saved under `data-bin/[my_data]`.
3. Run `bash scripts/mtcue/inference.sh --data-bin [my_data] --model [model_name]/[checkpoint_name].pt` for inference. Hypotheses will be saved in `data/evaluation/my_data/system.[tgt]`.

You can specify the input data and other inference parameters using command-line arguments or a configuration file.

### Pretrained Models

We provide pre-trained models for the experiments described in the paper. You can download the models from [Insert link to the pre-trained models here]. To use a pre-trained model, follow the instructions in the [README.md](README.md) file.


## Citation

If you use this code in your research, please cite our paper:

```
@misc{vincent2023mtcue,
      title={MTCue: Learning Zero-Shot Control of Extra-Textual Attributes by Leveraging Unstructured Context in Neural Machine Translation}, 
      author={Sebastian Vincent and Robert Flynn and Carolina Scarton},
      year={2023},
      eprint={2305.15904},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
(An official citation for ACL will be inserted here when the camera-ready version is out.)

## Contact

If you have any questions or issues with the code, please contact [Insert contact information here].
