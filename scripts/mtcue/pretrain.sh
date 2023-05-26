#!/bin/bash

set -e

seed=1

src=$1
tgt=$2
arch=$3
size=$4

model=$arch.$src.$tgt
ckpt=checkpoints/$model

mkdir -p $ckpt

if [ $size = big ]; then
	wc=0.0
	lr=0.001
	bsz=3584
	uf=64
else
	wc=0.0001
	lr=0.0005
	bsz=25000
	uf=1
fi

fairseq-train data-bin/baseline.$src.$tgt \
    --wandb-project mtcue_pretrain.$src.$tgt \
    --user-dir mtcue --task translation \
    --max-epoch 75 --patience 5 \
    --source-lang $src --target-lang $tgt \
    --arch $arch --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --dropout 0.3 --weight-decay $wc --clip-norm 1.0 \
    --lr $lr --lr-scheduler inverse_sqrt \
    --max-tokens $bsz --update-freq $uf \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --save-dir $ckpt --no-epoch-checkpoints \
    --memory-efficient-fp16 \
    --seed $seed
