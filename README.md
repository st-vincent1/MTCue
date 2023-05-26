# MTCue
## Learning Zero-Shot Control of Extra-Textual Attributes by Leveraging Unstructured Context in Neural Machine Translation

This repository contains the code used in the experiments described in the paper "MTCue: Learning Zero-Shot Control of Extra-Textual Attributes by Leveraging Unstructured Context in Neural Machine Translation" accepted to Findings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023). The code is written in Python and bash.

## Abstract

Efficient utilisation of both intra- and extra-textual context remains one of the critical gaps between machine and human translation. Existing research has primarily focused on providing individual, well-defined types of context in translation, such as the surrounding text or discrete external variables like the speaker's gender. This work introduces MTCue, a novel neural machine translation framework which interprets all context (including discrete variables) as text. MTCue learns an abstract representation of context, enabling transferability across different data settings and leveraging similar attributes in low-resource scenarios. With a focus on a dialogue domain with access to document and metadata context, we extensively evaluate MTCue in four language pairs in both translation directions. Our framework demonstrates significant improvements in translation quality over a parameter-matched non-contextual baseline, as measured by BLEU (+0.88) and Comet (+1.58). Moreover, MTCue significantly outperforms a "tagging" baseline at translating English text. Analysis reveals that the context encoder of MTCue learns a representation space that organises context based on specific attributes, such as formality, enabling effective zero-shot control. Pre-training on context embeddings also improves MTCue's few-shot performance compared to the "tagging" baseline. Finally, the ablation study conducted on model components and contextual variables further supports of the robustness of MTCue. 

## Requirements

- Python 3.10
- pip packages: fairseq, einops, sacrebleu

## Installation

To install the code in this repository, please follow the instructions in the [INSTALL.md](INSTALL.md) file.

## Usage

To use the code in this repository, please follow the instructions in the [USAGE.md](USAGE.md) file.

## Reproducing Experiments

To reproduce the experiments described in the paper, please follow the instructions in the [REPRODUCIBILITY.md](REPRODUCIBILITY.md) file.

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

If you have any questions or issues with the code, please raise an Issue in this Git repository or contact me directly at the e-mail address provided in the paper.
