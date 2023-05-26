#!/bin/bash

# This script evaluates the provided en-pl model on the EAMT22 test set.

set -e

while [ "$1" != "" ]; do
    case $1 in
        --model )               shift
                                arch="$1"
                                ;;
        --n )                   shift
                                n="$1"
                                ;;
        * )                     echo "Wrong args"
                                exit 1
    esac
    shift
done


if [ $arch = mtcue ]; then 
	cxt_model=minilm
elif [ $arch = tagging ]; then
	cxt_model=hash
elif [ $arch = novotney_cue ]; then
	cxt_model=distilbert
fi

if [ $n = max ]; then
	n=10000000
fi

# Assign model name
if [ $arch = baseline ]; then
  model=$arch.en.pl
  task=translation
  extras=""
  n=1
elif [ $n = 0 ]; then 
  model=mtcue.en.pl.next_os.$arch.$cxt_model.s
  task=contextual_translation
  extras="--context-model $cxt_model --no-doc-context"
  n=1
else
  model=eamt22.$n.$arch.$cxt_model 
  task=contextual_translation
  extras="--context-model $cxt_model --no-doc-context"
fi


test_ref=data/en-pl/eamt22/test.pl
valid_ref=data/en-pl/eamt22/dev.pl

out=data/hypotheses/eamt.$model

mkdir -p $out

for r in test valid; do
        fairseq-generate data-bin/eamt22.en.pl.$n \
            --gen-subset $r \
            --user-dir fairseq/examples/mtcue \
            --task $task --source-lang en --target-lang pl \
            --path checkpoints/$model/checkpoint_best.pt \
            --batch-size 256 $extras \
            --remove-bpe=sentencepiece > $out/$r.sys
    
        grep ^H $out/$r.sys | LC_ALL=C sort -V > $out/$r.hyp
        
	printf "EAMT22: Model [%s,%s] finetuned on %s samples yields " $arch $cxt_model $n
        if [ $r = valid ]; then
                sacrebleu $valid_ref -i $out/$r.hyp -m bleu -w 2 -b
        else
                sacrebleu $test_ref -i $out/$r.hyp -m bleu -w 2 -b
        fi
done

