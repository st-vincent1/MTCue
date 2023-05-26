#!/bin/bash

set -e
tgt=$1
arch=$2
emb=$3

model=mtcue.en.$tgt.next_os.$arch.$emb
data=data/evaluation/machine-translation-gender-eval/data/context

echo "Contextual"

#dest=$TMPDIR/$tgt.$arch.$emb
#mkdir -p $dest $dest/context
#rm -f $dest/context/test.*
## 1. contextual translation
#awk -F'<sep> ' '{print $1}' $data/geneval-context-wikiprofessions-2to1-test.en_$tgt.en > $dest/context/test.one.cxt
#awk -F'<sep> ' '{print $2}' $data/geneval-context-wikiprofessions-2to1-test.en_$tgt.en > $dest/tmp.en
#cat $data/geneval-context-wikiprofessions-original-test.en_$tgt.$tgt > $dest/tmp.$tgt
#awk -F'<sep> ' '{print $2}' $data/geneval-context-wikiprofessions-2to1-test.en_$tgt.en > $dest/context/test.zero.cxt
##yes "Feminine" | head -n 1100 > $dest/context/test.gender.cxt
#
#spm_encode=fairseq/scripts/spm_encode.py
#echo "encoding dev with learned BPE..."
#
#python ${spm_encode} \
#    --model training_data/baseline.en.$tgt/spm.bpe.model \
#    --output_format=piece \
#    --inputs $dest/tmp.en $dest/tmp.$tgt --outputs $dest/tmp.bpe.en $dest/tmp.bpe.$tgt
#
#data_bin=$TMPDIR/fairseq/data-bin/emnlp_gender
#rm -rf $data_bin
#fairseq-preprocess --joined-dictionary \
#    --source-lang en --target-lang $tgt \
#    --srcdict fairseq/data-bin/baseline.en.$tgt/dict.en.txt \
#    --testpref $dest/tmp.bpe \
#    --destdir $data_bin \
#    --workers 50
#
#rm -f $dest/test.$emb.{en.bin,cxt.bin,json}
#python context_embeddings/small_embeddings.py --paths $dest --dest_path $dest --model $emb --lang $tgt
#
##cp $dest/test.$emb.en.bin $data_bin/test.en-$tgt.$emb.en.bin
##cp $dest/test.$emb.en.idx $data_bin/test.en-$tgt.$emb.en.idx
#cp $dest/test.$emb.cxt.bin $data_bin/test.en-$tgt.$emb.cxt.bin
#cp $dest/test.$emb.cxt.idx $data_bin/test.en-$tgt.$emb.cxt.idx
#cp $dest/test.$emb.json $data_bin/test.en-$tgt.$emb.json
#
#fairseq-generate $data_bin \
#    --user-dir mtcue \
#    --task contextual_translation --source-lang en --target-lang $tgt \
#    --path fairseq/checkpoints/$model/checkpoint_best.pt \
#    --batch-size 128 \
#    --context-model $emb \
#    --no-doc-context \
#    --remove-bpe=sentencepiece > $dest/tmp.sys
#
#fairseq-generate $data_bin \
#    --task translation --source-lang en --target-lang $tgt \
#    --path fairseq/checkpoints/baseline.en.$tgt/checkpoint_best.pt \
#    --batch-size 128 \
#    --remove-bpe=sentencepiece > $dest/tmp.sys
#
#grep ^H $dest/tmp.sys | LC_ALL=C sort -V | cut -f3- > $dest/tmp.hyp
#
#sacrebleu $dest/tmp.$tgt -i $dest/tmp.hyp -m bleu chrf --chrf-word-order 2
#scorer=data/evaluation/machine-translation-gender-eval/accuracy_metric.py
#cor_refs=$data/geneval-context-wikiprofessions-original-test.en_$tgt.$tgt
#inc_refs=$data/geneval-context-wikiprofessions-flipped-test.en_$tgt.$tgt
## Call script with original and flipped.
#python $scorer --hypotheses $dest/tmp.hyp \
#               --cor_ref $cor_refs \
#               --inc_ref $inc_refs
#

echo "Counterfactual"
dest=$TMPDIR/$tgt.$arch.$emb
data=data/evaluation/machine-translation-gender-eval/data/sentences/test

mkdir -p $dest $dest/context
rm -f $dest/context/test.*
rm -f $dest/tmp*
# 1. contextual translation
for gen in masculine feminine; do
	for l in en $tgt; do
		cat $data/geneval-sentences-"$gen"-test.en_$tgt.$l >> $dest/tmp.$l
	done
done

yes "He is male" | head -n 300 > $dest/context/test.gender.cxt
yes "She is female" | head -n 300 >> $dest/context/test.gender.cxt

yes "Mr President" | head -n 300 > $dest/context/test.prof.cxt
yes "Madam President" | head -n 300 >> $dest/context/test.prof.cxt
#yes "Drama, Romance" | head -n 300 > $dest/context/test.gender.cxt
#yes "Drama, Romance" | head -n 300 >> $dest/context/test.gender.cxt


spm_encode=fairseq/scripts/spm_encode.py
echo "encoding dev with learned BPE..."

python ${spm_encode} \
    --model training_data/baseline.en.$tgt/spm.bpe.model \
    --output_format=piece \
    --inputs $dest/tmp.en $dest/tmp.$tgt --outputs $dest/tmp.bpe.en $dest/tmp.bpe.$tgt

data_bin=$TMPDIR/fairseq/data-bin/emnlp_gender
rm -rf $data_bin
fairseq-preprocess --joined-dictionary \
    --source-lang en --target-lang $tgt \
    --srcdict fairseq/data-bin/baseline.en.$tgt/dict.en.txt \
    --testpref $dest/tmp.bpe \
    --destdir $data_bin \
    --workers 50

rm -f $dest/test.$emb.{en.bin,en.cxt,cxt.bin,cxt.idx,json}
python context_embeddings/small_embeddings.py --paths $dest --dest_path $dest --model $emb --lang $tgt

#cp $dest/test.$emb.en.bin $data_bin/test.en-$tgt.$emb.en.bin
#cp $dest/test.$emb.en.idx $data_bin/test.en-$tgt.$emb.en.idx
cp $dest/test.$emb.cxt.bin $data_bin/test.en-$tgt.$emb.cxt.bin
cp $dest/test.$emb.cxt.idx $data_bin/test.en-$tgt.$emb.cxt.idx
cp $dest/test.$emb.json $data_bin/test.en-$tgt.$emb.json

fairseq-generate $data_bin \
    --user-dir mtcue \
    --task contextual_translation --source-lang en --target-lang $tgt \
    --path fairseq/checkpoints/$model/checkpoint_best.pt \
    --batch-size 128 \
    --context-model $emb \
    --no-doc-context \
    --remove-bpe=sentencepiece > $dest/tmp.sys

#fairseq-generate $data_bin \
#    --task translation --source-lang en --target-lang $tgt \
#    --path fairseq/checkpoints/baseline.en.$tgt/checkpoint_best.pt \
#    --batch-size 128 \
#    --remove-bpe=sentencepiece > $dest/tmp.sys


grep ^H $dest/tmp.sys | LC_ALL=C sort -V | cut -f3- > $dest/tmp.hyp

sacrebleu $dest/tmp.$tgt -i $dest/tmp.hyp -m bleu chrf --chrf-word-order 2
scorer=data/evaluation/machine-translation-gender-eval/accuracy_metric.py
head -n 300 $dest/tmp.hyp > $dest/masc.hyp
tail -n 300 $dest/tmp.hyp > $dest/fem.hyp

m_refs=$data/geneval-sentences-masculine-test.en_$tgt.$tgt
f_refs=$data/geneval-sentences-feminine-test.en_$tgt.$tgt

# Call script with original and flipped.
echo "Masculine"
python $scorer --hypotheses $dest/masc.hyp \
               --cor_ref $m_refs \
               --inc_ref $f_refs
echo "Feminine"
python $scorer --hypotheses $dest/fem.hyp \
               --cor_ref $f_refs \
               --inc_ref $m_refs
