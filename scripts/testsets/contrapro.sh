#!/bin/bash
set -e

model=$1
embedder=$2
opts=$3
tgt=$4

suffix=$suffix.$embedder
if [ $tgt = de ]; then
	cp=../test_suite_evaluation/ContraPro
else
	cp=../test_suite_evaluation/Large-contrastive-pronoun-testset-EN-FR/OpenSubs
fi
if [[ $opts = no_meta ]]; then
	extra="--no-meta-context"
elif [[ $opts = no_doc ]]; then
	extra="--no-doc-context"
else
	extra=""
fi
spm_encode=fairseq/scripts/spm_encode.py

data=data/evaluation/ContraPro/$tgt
for text in context.trg context.src current.src current.trg; do
	sed -i -e 's/^ *-* *-* *//g' $data/originals/$tgt.$text
done

python ${spm_encode} \
    --model training_data/baseline.en.$tgt/spm.bpe.model \
    --output_format=piece \
    --inputs $data/originals/$tgt.current.src $data/originals/$tgt.current.trg \
    --outputs $data/test.en-$tgt.bpe.en $data/test.en-$tgt.bpe.$tgt

data_bin=data-bin/contrapro_$tgt
if [ ! -d $data_bin ]; then
	fairseq-preprocess --joined-dictionary \
	    --source-lang en --target-lang $tgt \
	    --srcdict data-bin/baseline.en.$tgt/dict.en.txt \
	    --testpref $data/test.en-$tgt.bpe \
	    --destdir $data_bin \
	    --workers 20
fi
ls -larth $data

cat $data/originals/$tgt.current.src > $data/context/test.0.en
cat $data/originals/$tgt.context.src | awk 'NR%5==1' > $data/context/test.1.en
cat $data/originals/$tgt.context.src | awk 'NR%5==2' > $data/context/test.2.en
cat $data/originals/$tgt.context.src | awk 'NR%5==3' > $data/context/test.3.en
cat $data/originals/$tgt.context.src | awk 'NR%5==4' > $data/context/test.4.en
cat $data/originals/$tgt.context.src | awk 'NR%5==0' > $data/context/test.5.en

wc -l $data/context/*

if [ ! -f $data_bin/test.en-$tgt.$embedder.cxt.idx ]; then
	python cxt2vec/main.py --paths $data --dest_path $data/embeddings --model $embedder
	cp $data/embeddings/test.$embedder.cxt.bin $data_bin/test.en-$tgt.$embedder.cxt.bin
	cp $data/embeddings/test.$embedder.cxt.idx $data_bin/test.en-$tgt.$embedder.cxt.idx
	cp $data/embeddings/test.$embedder.en.bin $data_bin/test.en-$tgt.$embedder.en.bin
	cp $data/embeddings/test.$embedder.en.idx $data_bin/test.en-$tgt.$embedder.en.idx
	cp $data/embeddings/test.$embedder.json $data_bin/test.en-$tgt.$embedder.json
fi

#fairseq-generate $data_bin \
#    --user-dir mtcue \
#    --task translation --source-lang en --target-lang $tgt \
#    --path fairseq/checkpoints/baseline.en.$tgt/checkpoint_best.pt \
#    --batch-size 128 \
#    --score-reference \
#    --remove-bpe=sentencepiece > $data/test.en-$tgt.sys
#
#printf "\nBaseline scores: \n"
#grep ^H $data/test.en-$tgt.sys | LC_ALL=C sort -V | awk -F'\t' '{ print $2 }' > $data/test.en-$tgt.scores

#if [ $tgt = de ]; then
#	python $cp/evaluate.py --reference $cp/contrapro.json --scores $data/test.en-$tgt.scores --maximize
#
#else
#	python $cp/scripts/evaluate.py --reference $cp/testset-en-$tgt.json --scores $data/test.en-$tgt.scores --maximize
#fi

fairseq-generate $data_bin \
    --user-dir mtcue \
    --task contextual_translation --source-lang en --target-lang $tgt \
    --path checkpoints/mtcue.en.$tgt.next_os.$model/checkpoint_best.pt \
    --batch-size 128 \
    --score-reference \
    --context-model $embedder $extra \
    --remove-bpe=sentencepiece > $data/test.en-$tgt.sys 

printf "\nDocument-level scores: \n"

grep ^H $data/test.en-$tgt.sys | LC_ALL=C sort -V | awk -F'\t' '{ print $2 }' > $data/test.en-$tgt.scores

if [ $tgt = de ]; then
	python $cp/evaluate.py --reference $cp/contrapro.json --scores $data/test.en-$tgt.scores --maximize

else
	python $cp/scripts/evaluate.py --reference $cp/testset-en-$tgt.json --scores $data/test.en-$tgt.scores --maximize
fi
