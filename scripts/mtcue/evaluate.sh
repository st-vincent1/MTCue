#!/bin/bash

# This script evaluates models on the OpenSubtitles test set.

set -e

while [ "$1" != "" ]; do
    case $1 in
        --src )                 shift
                                src="$1"
                                ;;
        --tgt )                 shift
                                tgt="$1"
                                ;;
        --arch )                shift
                                arch="$1"
                                ;;
        --list-of-ablations )   shift
                                loa="$1"
                                ;; # Optional; can be used to ablate individual metadata
                                # e.g. to ablate plot and genre, use "--list-of-ablations plot,genre"
        --base )                base=1
                                ;; 
        * )                     echo "Wrong args"
                                exit 1
    esac
    shift
done
if [ $src = en ]; then data=data/$src-$tgt; else data=data/$tgt-$src; fi

if [ $arch = mtcue ]; then 
	cxt_model=minilm
elif [ $arch = tagging_100 ]; then
	cxt_model=hash
elif [ $arch = novotney_cue ]; then
	cxt_model=distilbert
fi

if [ -z $loa ]; then
        loa=noablations
fi

# Assign model name
if [ $base = 1 ]; then
  model=$arch.$src.$tgt
  task=translation
  extras=""
else
  model=mtcue.$src.$tgt.next_os.${arch}.${cxt_model}
  task=contextual_translation
  extras="--list-of-ablations $loa --context-model $cxt_model"
fi


out=data/hypotheses/os.$arch.$src.$tgt
mkdir -p $out

fairseq-generate data-bin/test.$src.$tgt \
    --user-dir mtcue \
    --task $task --source-lang $src --target-lang $tgt \
    --path checkpoints/$model/checkpoint_best.pt \
    --batch-size 512 $extras \
    --remove-bpe=sentencepiece > $out/test.sys

grep ^H $out/test.sys | LC_ALL=C sort -V | cut -f3- > $out/test.hyp

echo "--- OpenSubtitles test" >&2
printf "BLEU on the %s-%s pair with %s: " $src $tgt $arch
sacrebleu $data/test.$tgt -i $out/test.hyp -m bleu -w 2 -b
