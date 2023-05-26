#!/bin/bash


set -e

while [ "$1" != "" ]; do
    case $1 in
        --src )                 shift
                                src="$1"
                                ;;
        --tgt )                 shift
                                tgt="$1"
                                ;;
        * )                     echo "Wrong args"
                                exit 1
    esac
    shift
done


tmp=$TMPDIR/mtcue.$src.$tgt # replace with whatever directory is suitable for you.
# This step is recommended since en-x and x-en pairs use the same directory, and files can be confused

mkdir -p $tmp

if [ $src = en ]; then
  pairname=$src-$tgt
else
  pairname=$tgt-$src
fi
dest_p=$src-$tgt

dest=data-bin/mtcue.$src.$tgt

spm_encode=fairseq/scripts/spm_encode.py
tgtdict=spm_dicts/$src.$tgt/fairseq.dict.$tgt.txt
spm_model=spm_dicts/$src.$tgt/spm.bpe.model

cp data/$pairname/{train,dev,test}.{$src,$tgt} $tmp
echo "-- TMP after merging $tmp" >&2

# Encode

for split in train dev test; do
        if [ ! -f $tmp/$split.bpe.$tgt ]; then
          echo "encoding train with learned BPE..."
          python $spm_encode \
            --model $spm_model \
            --output_format=piece \
            --inputs $tmp/$split.$src $tmp/$split.$tgt \
            --outputs $tmp/$split.bpe.$src $tmp/$split.bpe.$tgt
        fi
done


if [ ! -d $dest ]; then
        fairseq-preprocess --joined-dictionary \
            --source-lang $src --target-lang $tgt \
            --tgtdict $tgtdict \
            --trainpref $tmp/train.bpe --validpref $tmp/dev.bpe --testpref $tmp/test.bpe \
            --destdir $dest \
            --workers 30
fi

for emb in minilm; do # can add "distilbert" or "hash" if running baselines
        for meta in cxt.bin $src.bin cxt.idx $src.idx json; do
                if [ ! -d data/$pairname/embeddings_$src ]; then
			python cxt2vec/main.py --model $emb --paths data/$pairname --dest_path data/$pairname/embeddings_$src --suffixes cxt,$src
		fi
                cp data/$pairname/embeddings_$src/train.$emb.$meta $dest/train.$dest_p.$emb.$meta
                cp data/$pairname/embeddings_$src/dev.$emb.$meta $dest/valid.$dest_p.$emb.$meta
                cp data/$pairname/embeddings_$src/test.$emb.$meta $dest/test.$dest_p.$emb.$meta
        done
done

rm -r $tmp

