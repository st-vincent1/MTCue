#!/bin/bash

# This script evaluates models on the OpenSubtitles test set.
# It also evaluates the EN-PL pair on the EAMT22 test set.
set -e

while [ "$1" != "" ]; do
    case $1 in
        --src )                 shift
                                src="$1"
                                ;;
        --tgt )                 shift
                                tgt="$1"
                                ;;
        --suffix )              shift 
                                suffix="$1"
                                ;;
        --arch )                shift
                                arch="$1"
                                ;;
        --list-of-ablations )   shift
                                loa="$1"
                                ;;
        --context-model )       shift
                                cxt_model="$1"
                                ;;
        --baseline )            baseline=1
                                ;;
        * )                     echo "Wrong args"
                                exit 1
    esac
    shift
done
if [ $src = en ]; then
	data=data/evaluation/zoo/$src-$tgt
else
	data=data/evaluation/zoo/$tgt-$src
fi

if [ -z $loa ]; then
	loa=noablations
fi
# Assign model name
if [ $arch = baseline ] || [ $arch = big_baseline ]; then
  model=$arch.$src.$tgt
  task=translation
  extras=""
else
  model=mtcue.$src.$tgt.$suffix.${arch}.${cxt_model}
  task=contextual_translation
  extras="--list-of-ablations $loa --context-model $cxt_model"
fi

cp -r fairseq/data-bin/zootest.$src.$tgt $TMPDIR/

out=data/hypotheses/$model
mkdir -p $out
tmp=${TMPDIR}/zootest.$src.$tgt

if [ $baseline = 1 ]; then
	#fairseq-generate $tmp \
	#    --user-dir mtcue \
	#    --task translation --source-lang $src --target-lang $tgt \
	#    --path fairseq/checkpoints/baseline.$src.$tgt/checkpoint_best.pt \
	#    --batch-size 128 \
	#    --remove-bpe=sentencepiece > $tmp/base.sys
	#grep ^H $tmp/base.sys | LC_ALL=C sort -V | cut -f3- > $out/base.hyp
	echo "--- ZOO test: baseline" >&2
	sacrebleu $data/test.$tgt -i $out/base.hyp -m bleu chrf --chrf-word-order 2
fi

fairseq-generate $tmp \
    --user-dir mtcue \
    --task $task --source-lang $src --target-lang $tgt \
    --path fairseq/checkpoints/$model/checkpoint_best.pt \
    --batch-size 128 $extras \
    --remove-bpe=sentencepiece > $tmp/test.sys

grep ^H $tmp/test.sys | LC_ALL=C sort -V | cut -f3- > $out/zoo.hyp
echo "--- ZOO test: MTCue" >&2
sacrebleu $data/test.$tgt -i $out/zoo.hyp -m bleu chrf --chrf-word-order 2
