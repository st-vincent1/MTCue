#!/bin/bash

set -e

id=$1
data=data/en-fr

model=hs_sweep.$id.mtcue.en.fr.next_os.mtcue_merger.sentence-transformers

cp -r fairseq/data-bin/test.en.fr $TMPDIR/
tmp=${TMPDIR}/test.en.fr

fairseq-generate $tmp \
    --user-dir mtcue \
    --task contextual_translation --source-lang en --target-lang fr \
    --path fairseq/checkpoints/$model/checkpoint_best.pt \
    --batch-size 128 \
    --list-of-ablations scene_prefix.fr,previous.fr,previous.en,current.fr \
    --remove-bpe=sentencepiece > $tmp/test.sys

out=data/hypotheses/hs_$id
mkdir -p $out
grep ^H $tmp/test.sys | LC_ALL=C sort -V | cut -f3- > $out/test.hyp
