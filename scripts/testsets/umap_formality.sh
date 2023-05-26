#!/bin/bash
set -e

suffix=$1
model=$2
suffix=$suffix.$model
ckpt="${3:-best}"
printf "Model is $model\n\n"

data=data/evaluation/IWSLT2022/data/test/en-de
TMPDIR=/tmp/form
mkdir -p $TMPDIR

rm -f $TMPDIR/context/test.*
mkdir -p $TMPDIR/context
> $TMPDIR/tmp.en
> $TMPDIR/tmp.de

prompt_file=data/en-de/formality_exps/zoo.meta.cxt

cat $prompt_file | while read line; do
	yes $line | head -n 600 >> $TMPDIR/context/test.formality.cxt
        head -n 600 $data/test.en-de.en >> $TMPDIR/tmp.en
        head -n 600 $data/test.en-de.de >> $TMPDIR/tmp.de
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
    --batch-size 128 \
    --context-model $model \
    --no-doc-context \
    --remove-bpe=sentencepiece > $TMPDIR/tmp.sys

grep ^H $TMPDIR/tmp.sys | LC_ALL=C sort -V | cut -f3- > $TMPDIR/tmp.hyp

sacrebleu $TMPDIR/tmp.de -i $TMPDIR/tmp.hyp -m bleu chrf --chrf-word-order 2
scorer=data/evaluation/IWSLT2022/scorer.py

f_r=$TMPDIR/formal_results
rm -f $f_r 

cat $prompt_file | while read line; do
	head -n 600 $TMPDIR/tmp.hyp > $TMPDIR/test.en-de.formal.hyp
	sed -i '1,600d' $TMPDIR/tmp.hyp

	printf "Calculating scores for FORMAL prompt (formal accuracy matters)\nPrompt is %s\n" "$line"
	python $scorer -hyp $TMPDIR/test.en-de.formal.hyp \
	-f $data/formality-control.test.en-de.formal.annotated.de \
	-if $data/formality-control.test.en-de.informal.annotated.de >> $f_r
	tail -n1 $f_r

  printf "\n\n\n\n"
done
# Get prompt-wide results
grep Formal $f_r | awk -F' |,' '{ sum += $3; n++ } END { print "Formal Acc: " sum / n }' > $TMPDIR/acc
awk '{sum += $3; n++} END {print "Total Acc: " sum / n}' $TMPDIR/acc >> $TMPDIR/acc

cat $TMPDIR/acc
cp $TMPDIR/formal_results data/en-de/formality_exps/zoo.accs
