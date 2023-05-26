#!/bin/bash

# Script used to fine-tune MTCue from a translation checkpoint

set -e

seed=1

while [ "$1" != "" ]; do
    case $1 in
        --src )                 shift
                                src="$1"
                                ;;
        --tgt )                 shift
                                tgt="$1"
                                ;;
        --extra )               shift
                                extra="$1"
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
        --from-model )          shift
                                from_model="$1"
                                ;;
        --big )                 big=1
                                ;; # Use --big to train MTCue-big

        * )                     echo "Wrong args"
                                exit 1
    esac
    shift
done

bsz=25000
warmup=4000

# Size-specific parameters
if [ $big = 1 ]; then
	lr=0.0001
	bsz=10000
	wc=0.0
else
	lr=0.0003
	bsz=25000
	wc=0.0001
fi



if [[ ! $loa = "" ]]; then
        echo "Ablating $loa"
else
        loa="placeholder"
fi


load_data_from=mtcue.$src.$tgt.next_os
model=mtcue.$src.$tgt.next_os$extra.$arch.$cxt_model


# We observed that on some Lustre storages the memory-mapped files with cxt embeddings load pretty slow compared to a standard machine
# If you experience similar issues, the fix is to transfer all files necessary to training to a drive more suites to many operations
# on small files - the code below should help; set $TMPDIR to your storage

scratch=$TMPDIR/data-bin/$load_data_from
mkdir -p $scratch
rsync -av --ignore-existing data-bin/$load_data_from/{train,valid}.$src-$tgt.{$src,$tgt}.{bin,idx} $scratch
rsync -av --ignore-existing data-bin/$load_data_from/dict* $scratch
rsync -av --ignore-existing data-bin/$load_data_from/*$cxt_model* $scratch

# Useful if we want to further fine-tune MTCue on another dataset
if [[ ! $from_model = "" ]]; then
        extras="--finetune-from-model checkpoints/$from_model/checkpoint_best.pt"
fi

# Turning off meta or doc context; needs to be done if either type is not provided
if [[ $extra = "_nometa" ]]; then
        extras="$extras --no-meta-context"
elif [[ $extra = "_nodoc" ]]; then
        extras="$extras --no-doc-context"
fi

# Select the correct model to pre-load
if [[ $big = 1 ]]; then
	preload=transformer_vaswani_wmt_en_de_big.baseline.$src.$tgt
else
	preload=baseline.$src.$tgt
fi

# Tagging loads a different model
if [[ $arch = tagging_pm ]]; then
	preload=mtcue_pm_pretrain.$src.$tgt
elif [[ $arch = tagging_100 ]]; then
	#preload=mtcue_100_pretrain.$src.$tgt
fi

fairseq-train $scratch \
    --wandb-project mtcue.$src.$tgt \
    --user-dir mtcue --task contextual_translation \
    --load-pretrained-weights checkpoints/$preload/checkpoint_best.pt \
    --context-model $cxt_model \
    --max-epoch 30 --patience 10 \
    --list-of-ablations $loa \
    --arch $arch --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr $lr --lr-scheduler inverse_sqrt \
    --max-tokens $bsz --update-freq 8 \
    --warmup-updates $warmup --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --weight-decay $wc --clip-norm 1.0 \
    --save-dir checkpoints/$model \
    --memory-efficient-fp16 --context-dropout 0.25 \
    --seed $seed $extras
