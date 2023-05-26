# Installation Guide for MTCue

This repository contains the code used in the experiments described in the paper "MTCue: Learning Zero-Shot Control of Extra-Textual Attributes by Leveraging Unstructured Context in Neural Machine Translation" accepted to Findings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023). The code is written in Python and bash.

## System Requirements

To use the code in this repository, your system should have the following:

- Python 3.10
- pip packages: fairseq, einops, sacrebleu

## Installation

To install the code in this repository, please follow these steps:

1. Clone this repository using the following command: 

   ```bash
   git clone --recurse-submodules https://github.com/st-vincent1/MTCue.git
   ```
   
2. Install the required dependencies. We recommend using a virtual environment. If you are using [Python], you can create a virtual environment using [venv](https://docs.python.org/3/tutorial/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Activate the virtual environment and install the required packages using the following command:

   ```bash
   conda install -c conda-forge gxx_linux-64
   pip install --editable fairseq
   pip install einops sacrebleu sentence-transformers
   ```

## Preparing data

The datasets can be downloaded under this link: ` `. Once downloaded, please unpack the `.zip` to `/path/to/this/repo/data`. To make data binaries necessary to train/evaluate the models, run

```bash
bash scripts/mtcue/prepare.sh --src <source language> --tgt <target language>
```
 
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
