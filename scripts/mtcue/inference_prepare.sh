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
        --data )                shift
                                data="$1" # folder name for your data in data/evaluation. Include only the folder name
                                ;;
        * )                     echo "Wrong args"
                                exit 1
    esac
    shift
done


dest=data-bin/$data

dedata=data/evaluation/$data

spm_encode=fairseq/scripts/spm_encode.py
tgtdict=spm_dicts/$src.$tgt/fairseq.dict.$tgt.txt
spm_model=spm_dicts/$src.$tgt/spm.bpe.model

if [ ! -f $data/test.bpe.$tgt ]; then
	echo "encoding train with learned BPE..."
        python $spm_encode \
            --model $spm_model \
            --output_format=piece \
            --inputs $dedata/test.$src $dedata/test.$tgt \
            --outputs $dedata/test.bpe.$src $dedata/test.bpe.$tgt
fi


if [ ! -d $dest ]; then
        fairseq-preprocess --joined-dictionary \
            --source-lang $src --target-lang $tgt \
            --tgtdict $tgtdict \
            --testpref $dedata/test.bpe \
            --destdir $dest \
            --workers 30
fi

for meta in cxt.bin $src.bin cxt.idx $src.idx json; do
        if [ ! -d data/$pairname/embeddings_$src ]; then
		python cxt2vec/main.py --model minilm --path $dedata --dest_path $dedata/embeddings_$src --suffixes cxt,$src
	fi
        cp $dedata/embeddings_$src/train.minilm.$meta $dest/train.$dest_p.minilm.$meta
        cp $dedata/embeddings_$src/dev.minilm.$meta $dest/valid.$dest_p.minilm.$meta
        cp $dedata/embeddings_$src/test.minilm.$meta $dest/test.$dest_p.minilm.$meta
done
