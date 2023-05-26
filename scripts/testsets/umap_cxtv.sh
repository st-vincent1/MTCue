#!/bin/bash
set -e

suffix=$1
model=$2
suffix=$suffix.$model
ckpt="${3:-best}"
printf "Model is $model\n\n"

data=data/evaluation/IWSLT2022/data/test/en-de

rm -f $TMPDIR/context/test.*
mkdir -p $TMPDIR/context
> $TMPDIR/tmp.en
> $TMPDIR/tmp.de
prompt_file=data/en-de/formality_exps/zoo.meta.cxt

cat $prompt_file | while read line; do
	echo $line >> $TMPDIR/context/test.formality.cxt
        echo "Here is a source." >> $TMPDIR/tmp.en
        echo "Hier ist eine Quelle." >> $TMPDIR/tmp.de
done

spm_encode=fairseq/scripts/spm_encode.py
echo "encoding dev with learned BPE..."

python ${spm_encode} \
    --model training_data/baseline.en.de/spm.bpe.model \
    --output_format=piece \
    --inputs $TMPDIR/tmp.en $TMPDIR/tmp.de --outputs $TMPDIR/tmp.bpe.en $TMPDIR/tmp.bpe.de

data_bin=$TMPDIR/fairseq/data-bin/iwslt22
rm -rf $data_bin
fairseq-preprocess --joined-dictionary \
    --source-lang en --target-lang de \
    --srcdict fairseq/data-bin/baseline.en.de/dict.en.txt \
    --testpref $TMPDIR/tmp.bpe \
    --destdir $data_bin \
    --workers 20

rm -f $TMPDIR/test.$model.{en.bin,cxt.bin,json}
python context_embeddings/small_embeddings.py --paths $TMPDIR --dest_path $TMPDIR --model $model --lang de

cp $TMPDIR/test.$model.cxt.bin $data_bin/test.en-de.$model.cxt.bin
cp $TMPDIR/test.$model.cxt.idx $data_bin/test.en-de.$model.cxt.idx
cp $TMPDIR/test.$model.json $data_bin/test.en-de.$model.json

ckpt_name=mtcue.en.de.$suffix	
#ckpt_name=$suffix
fairseq-generate $data_bin \
    --user-dir mtcue \
    --task contextual_translation --source-lang en --target-lang de \
    --path fairseq/checkpoints/$ckpt_name/checkpoint_$ckpt.pt \
    --batch-size 256 --beam 1 \
    --context-model $model \
    --no-doc-context \
    --remove-bpe=sentencepiece 
