#!/bin/bash

set -e

tmp=$TMPDIR/eamt
mkdir -p $tmp

spm_encode=fairseq/scripts/spm_encode.py
tgtdict=data-bin/baseline.en.pl/dict.pl.txt
spm_model=training_data/baseline.en.pl/spm.bpe.model


data=data/en-pl/eamt22
dest=data-bin/eamt22.en.pl

# Encode with spm
for split in train dev test; do
	if [ ! -f $data/1/$split.bpe.en ]; then
		python $spm_encode \
	    	  --model $spm_model \
		  --output_format=piece \
	    	  --inputs $data/1/$split.en $data/1/$split.pl \
	    	  --outputs $data/1/$split.bpe.en $data/1/$split.bpe.pl
	fi
done

for n in 1 5 30 300 3000 30000 10000000; do
	n_dest="$dest".$n
	if [ ! -d $n_dest ]; then
		fairseq-preprocess --joined-dictionary --tgtdict $tgtdict \
		  --source-lang en --target-lang pl \
		  --trainpref $data/1/train.bpe --validpref $data/1/dev.bpe --testpref $data/1/test.bpe \
		  --destdir $n_dest --workers 50
	fi
        #if [ ! -d $data/$n/$embeddings ]; then
	python cxt2vec/main.py --paths $data/$n --dest_path $data/$n/embeddings --suffixes cxt --model minilm 
	#fi
	# Copy embeddings of context into data-bins
	for meta in cxt.bin cxt.idx json; do
	  for model in minilm; do
	      cp $data/$n/embeddings/train.$model.$meta $n_dest/train.en-pl.$model.$meta
	      cp $data/$n/embeddings/dev.$model.$meta $n_dest/valid.en-pl.$model.$meta
	      cp $data/$n/embeddings/test.$model.$meta $n_dest/test.en-pl.$model.$meta
	  done
	done
done
