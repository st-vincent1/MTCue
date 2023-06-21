# Reproducibility Guide for MTCue

This guide explains how to reproduce the results of the experiments described in the paper. Before reproducing the experiments, please make sure you have installed the necessary dependencies as described in the [INSTALL.md](INSTALL.md) file and have downloaded any necessary datasets or models. Note that this guide focuses on reproducing the evaluation results from already trained models, and if you wish to train the models yourself beforehand please head to [USAGE.md](USAGE.md). Furthermore, due to limited storage we only share checkpoints of Base and MTCue models (Novotney-CUE and Tagging baseline checkpoints are available upon request).

## Translation quality

To train or evaluate models for a specific language pair, replace `[SRC]` and `[TGT]` with the code of the source/target languages [en/fr/de/ru]. For example, to run the English-to-Russian language pair, use `en` and `ru`.

### Dataset

To reproduce the experiments, extract the OpenSubtitles18 dataset following the instructions in [this repo](https://github.com/st-vincent1/opensubtitles_parser), then move the relevant directories to the `data/` directory under this repo.

**Note**: To extract the film metadata for training, you will need to provide an API key for [the OMDb API](https://www.patreon.com/join/omdb). You can obtain one by becoming a Patron with a [Basic subscription bought here](https://www.patreon.com/join/omdb). 

### Checkpoints

The checkpoints for MTCue can be downloaded under [this FigShare link](https://figshare.shef.ac.uk/articles/dataset/MTCue_Model_Checkpoints/22956002/1). Checkpoints are sorted by language pair, i.e. checkpoints for English-to-German are located in `en.de.zip`. Unpack the relevant checkpoints to the `checkpoints/` directory:
1. Navigate to the this repository's path.
2. Download the checkpoints here or move them here from your download location.
3. Replace [SRC] and [TGT] with the source and language codes respectively and run:
```bash
unzip [SRC].[TGT].zip
mv [SRC].[TGT]/* checkpoints
ls checkpoints
```
This should yield
```bash
(...) baseline.[SRC].[TGT] mtcue.[SRC].[TGT] (...)
```

### Evaluation

To evaluate the relevant, run the following command:

```
bash scripts/mtcue/evaluate.sh --src [SRC] --tgt [TGT] --model [tagging/novotney_cue/mtcue]
```

### Results

The results should match Table 4 in the paper:

![image](https://github.com/st-vincent1/MTCue/assets/19303946/9808ae0e-19a6-4b99-a9c3-85e80e501404)

## IWSLT22 Formality control task (English-to-German, English-to-Russian)

### Dataset

To reproduce the results on the IWSLT22 Formality Control task, follow the instructions above to train/download MTCue models (no specific training is necessary for this test set). 

Then clone the task repository anywhere and copy the IWSLT22 folder to `data/evaluation/`:

```bash
git clone https://github.com/amazon-science/contrastive-controlled-mt
mkdir -p /path/to/this/repository/data/evaluation/IWSLT22
mv contrastive-controlled-mt/IWSLT22/* /path/to/this/repository/data/evaluation/IWSLT22
rm -r contrastive-controlled-mt
```

Finally, run evaluation with

```bash
bash scripts/testsets/formality.sh --lang [de/ru] --arch mtcue
```

### Results

The results should match the table below (Table 5 in the paper):

<img src="https://github.com/st-vincent1/MTCue/assets/19303946/f0ef4591-4dd9-414a-93e9-4eaedaeba20c" width="480">


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

If you have any questions or issues with the code, please contact me at `stvincent1@sheffield.ac.uk` or alternatively raise an Issue/make a Pull Request within this repository.
