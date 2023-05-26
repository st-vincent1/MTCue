#!/bin/bash

# This script can be used for inference with a custom model and dataset.

set -e

while [ "$1" != "" ]; do
    case $1 in
        --src )                 shift
                                src="$1"
                                ;;
        --tgt )                 shift
                                tgt="$1"
                                ;;
        --data-bin )            shift
                                data_bin="$1" # the name of the data-bin folder, e.g. my_data
                                ;;
        --model )               shift
                                model="$1" # checkpoint name including folder name, e.g. my_model/checkpoint.pt
                                ;;
        * )                     echo "Wrong args"
                                exit 1
    esac
    shift
done

out=data/hypotheses/$data_bin
mkdir -p $out

fairseq-generate data-bin/$data_bin \
    --user-dir mtcue \
    --task $task --source-lang $src --target-lang $tgt \
    --path checkpoints/$model \
    --batch-size 512 $extras \
    --remove-bpe=sentencepiece > $out/test.sys

grep ^H $out/test.sys | LC_ALL=C sort -V | cut -f3- > $out/system.$tgt

# If reference file exists, compute BLEU with the script below, replacing REF with the filepath
# printf "BLEU on the %s-%s pair with %s: " $src $tgt $model
# sacrebleu REF -i $out/test.hyp -m bleu -w 2 -b
