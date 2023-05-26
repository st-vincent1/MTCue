# Reproducibility Guide for MTCue

This guide explains how to reproduce the results of the experiments described in the paper. Before reproducing the experiments, please make sure you have installed the necessary dependencies as described in the [INSTALL.md](INSTALL.md) file and have downloaded any necessary datasets or models. Note that this guide focuses on reproducing the evaluation results from already trained models, and if you wish to train the models yourself beforehand please head to [USAGE.md](USAGE.md). Furthermore, due to limited storage we only share checkpoints of Base and MTCue models (Novotney-CUE and Tagging baseline checkpoints are available upon request).

## Translation quality

To train or evaluate models for a specific language pair, replace `[SRC]` and `[TGT]` with the code of the source/target languages [en/fr/de/ru]. For example, to run the English-to-Russian language pair, use `en` and `ru`.

### Dataset

To reproduce the experiments, download the OpenSubtitles18 dataset from [link to dataset] and extract it to the `data/` directory.

### Checkpoints

Unpack the relevant checkpoints to the `checkpoints/` directory.

### Evaluation

To evaluate the relevant, run the following command:

```
bash scripts/mtcue/evaluate.sh --src [SRC] --tgt [TGT] --model [tagging/novotney_cue/mtcue]
```

### Results

The results should match Table 3 in the paper.


## EAMT22 English-to-Polish multi-attribute control task

### Dataset

To reproduce the experiments for the EAMT22 task, download the EAMT22 dataset from [link to dataset] and extract it to the `data/en-pl/eamt22/` directory.

### Checkpoints

The relevant checkpoints are located under `en-pl/eamt22/checkpoints/`. Unpack them to `checkpoints/`.

### Evaluation

To evaluate the model for the EAMT22 task, run the following command:

```bash
bash scripts/testsets/eamt/evaluate.sh --n [0/1/5/30/300/3000/30000/max] --model mtcue
```
`n=0` corresponds to the zero-shot scenario.

The above script will generate hypotheses under `data/hypotheses/` and calculate BLEU.

To obtain the accuracy scores, follow the instructions below to install the evaluation tool.

#### Installing the evaluation tool

1. Download the Morfeusz model:
```bash
wget http://zil.ipipan.waw.pl/SpacyPL?action=AttachFile&do=get&target=pl_spacy_model_morfeusz_big-0.1.0.tar.gz
```
2. Install spacy via `pip install spacy=2.2.4`

#### Running the evaluation tool 

To run the evaluation tool, run the following command:
```bash
python eamt_annotation_tool/eamt22_evaluate.py --hyp data/hypotheses/eamt.eamt22.[n].mtcue/test.hyp
```

You can specify the path to the test set and other evaluation parameters using command-line arguments or a configuration file.

### Results

[Insert a summary of the results for Experiment 2 here]


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

[Insert a summary of the results for Experiment 2 here]


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