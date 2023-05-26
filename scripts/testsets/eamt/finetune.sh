#!/bin/bash

set -e

seed=1

while [ "$1" != "" ]; do
    case $1 in
        --n )                   shift
                                n="$1"
                                ;;
        --context-model )       shift
                                cxt_model="$1"
                                ;;
        --from-model )          shift
                                from_model="$1"
                                ;;
        * )                     echo "Wrong args"
                                exit 1
    esac
    shift
done

if [[ $from_model != none ]]; then
	model=mtcue.en.pl.next_os.$from_model.$cxt_model
	extras="--finetune-from-model checkpoints/$model/checkpoint_best.pt"
else
	from_model=tagging_big
fi


scratch=$TMPDIR/data-bin/$load_data_from.$suffix

mkdir -p $scratch

rsync -av --ignore-existing data-bin/eamt22.en.pl.$n/{train,valid}.en-pl.{en,pl}.{bin,idx} $scratch
rsync -av --ignore-existing data-bin/eamt22.en.pl.$n/dict* $scratch
rsync -av --ignore-existing data-bin/eamt22.en.pl.$n/*$cxt_model* $scratch

if [ $from_model = mtcue_big ]; then
	lr=0.0001
	bsz=10000
	wc=0.0
else
	lr=0.0003
	bsz=15000
	wc=0.0001
fi

echo $extras
fairseq-train $scratch \
    --wandb-project eamt22 \
    --user-dir fairseq/examples/mtcue --task contextual_translation \
    --context-model $cxt_model \ 
    --load-pretrained-weights checkpoints/big_baseline.en.pl/checkpoint_best.pt \
    --max-epoch 20 --patience 5 \
    --arch $from_model --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr $lr --lr-scheduler inverse_sqrt \
    --max-tokens $bsz --update-freq 8 \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --weight-decay $wc --clip-norm 1.0 \
    --save-dir checkpoints/eamt22.$n.$from_model.$cxt_model \
    --memory-efficient-fp16 \
    --seed $seed --no-doc-context $extras
