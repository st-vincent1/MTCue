#!/bin/bash

# This script evaluates models on the OpenSubtitles dev set.
# It also evaluates the EN-PL pair on the EAMT22 dev set.
set -e

while [ "$1" != "" ]; do
    case $1 in
	--src )			shift
				src="$1"
				;;
	--tgt )			shift
				tgt="$1"
				;;
        --id )                  shift 
                                id="$1"
                                ;;
        * )                     echo "Wrong args"
                                exit 1
    esac
    shift
done

if [ $src = en ]; then
	data=data/$src-$tgt
else
	data=data/$tgt-$src
fi

model=hs_sweep.$id.$src.$tgt
task=contextual_translation
cp -r fairseq/data-bin/valid.$src.$tgt $TMPDIR/
tmp=${TMPDIR}/valid.$src.$tgt
rename $tmp/valid $tmp/test $tmp/valid* 

for loa in none writers,genre,year_of_release,rated,country,previous.$src,previous.$tgt,scene_prefix.$tgt,current.$src; do
	extras="--list-of-ablations $loa" 

	fairseq-generate $tmp \
	    --user-dir mtcue \
	    --task $task --source-lang $src --target-lang $tgt \
	    --path fairseq/checkpoints/$model/checkpoint_best.pt \
	    --batch-size 128 $extras \
	    --remove-bpe=sentencepiece > $tmp/dev.sys

	out=data/hypotheses/hs2
        mkdir -p $out 
	grep ^H $tmp/dev.sys | LC_ALL=C sort -V | cut -f3- > $out/$loa.$id.$src.$tgt.hyp
	echo "--- OpenSubtitles dev" >&2
	sacrebleu $data/dev.$tgt -i $out/$loa.$id.$src.$tgt.hyp -m chrf --chrf-word-order 2
done
