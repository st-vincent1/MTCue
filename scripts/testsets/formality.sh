#!/bin/bash
set -e

while [ "$1" != "" ]; do
    case $1 in
        --lang )                shift
                                lang="$1"
                                ;;
        --arch )                shift
                                arch="$1"
                                ;;
        * )                     echo "Wrong args"
                                exit 1
    esac
    shift
done

cxt_model=minilm

if [ $arch = mtcue ]; then 
	cxt_model=minilm
elif [ $arch = tagging ]; then
	cxt_model=hash
elif [ $arch = novotney_cue ]; then
	cxt_model=distilbert
fi
suffix=next_os.$arch.$cxt_model.actual_s.15000


printf "Model is $cxt_model\n\n"

data=data/evaluation/IWSLT2022/data/test/en-$lang

load_from=$data/formality-control.test
bitext=$data/test.en-$lang

# paste english twice
> ${bitext}.en
cat $load_from.en-$lang.en >> $bitext.en
cat $load_from.en-$lang.en >> $bitext.en
# put together formal/informal refs
> $bitext.$lang
cat $load_from.en-$lang.informal.$lang >> $bitext.$lang
cat $load_from.en-$lang.formal.$lang >> $bitext.$lang
wc -l ${bitext}.$lang
wc -l ${bitext}.en

declare -a formal_prompts=("Formal conversation"
                           "Hannah Larsen, meet Sonia Jimenez. One of my most favourite nurses."
                           "In case anything goes down we need all the manpower alert, not comfortably numb."
                           "Biography, Drama"
                           "A musician travels a great distance to return an instrument to his elderly teacher")
declare -a informal_prompts=("Informal chit-chat"
                             "Animation, Adventure, Comedy"
                             "Drama, Family, Romance"
                             "I'm gay for Jamie."
                             "What else can a pathetic loser do?")

rm -f $TMPDIR/context/test.*
mkdir -p $TMPDIR/context
> $TMPDIR/tmp.en
> $TMPDIR/tmp.$lang
for index in "${!formal_prompts[@]}"; do
        f_p="${formal_prompts[$index]}"
        i_p="${informal_prompts[$index]}"

        yes $f_p | head -n 600 >> $TMPDIR/context/test.formality.cxt
        yes $i_p | head -n 600 >> $TMPDIR/context/test.formality.cxt
        cat $data/test.en-$lang.en >> $TMPDIR/tmp.en
        cat $data/test.en-$lang.$lang >> $TMPDIR/tmp.$lang
done

python fairseq/scripts/spm_encode.py \
    --model training_data/baseline.en.$lang/spm.bpe.model --output_format=piece \
    --inputs $TMPDIR/tmp.en $TMPDIR/tmp.$lang --outputs $TMPDIR/tmp.bpe.en $TMPDIR/tmp.bpe.$lang

data_bin=$TMPDIR/data-bin/iwslt22
rm -rf $data_bin
fairseq-preprocess --joined-dictionary \
    --source-lang en --target-lang $lang \
    --srcdict data-bin/baseline.en.$lang/dict.en.txt \
    --testpref $TMPDIR/tmp.bpe \
    --destdir $data_bin \
    --workers 20

rm -f $TMPDIR/test.$cxt_model.{en.bin,cxt.bin,json}
python cxt2vec/main.py --path $TMPDIR --dest_path $TMPDIR --model $cxt_model --suffixes cxt

cp $TMPDIR/test.$cxt_model.cxt.bin $data_bin/test.en-$lang.$cxt_model.cxt.bin
cp $TMPDIR/test.$cxt_model.cxt.idx $data_bin/test.en-$lang.$cxt_model.cxt.idx
cp $TMPDIR/test.$cxt_model.json $data_bin/test.en-$lang.$cxt_model.json

ckpt_name=mtcue.en.$lang.$suffix

fairseq-generate $data_bin \
    --user-dir mtcue \
    --task contextual_translation --source-lang en --target-lang $lang \
    --path checkpoints/$ckpt_name/checkpoint_last.pt \
    --batch-size 256 \
    --context-model $cxt_model \
    --no-doc-context \
    --remove-bpe=sentencepiece > $TMPDIR/tmp.sys

grep ^H $TMPDIR/tmp.sys | LC_ALL=C sort -V | cut -f3- > $TMPDIR/tmp.hyp

sacrebleu $TMPDIR/tmp.$lang -i $TMPDIR/tmp.hyp -m bleu chrf --chrf-word-order 2
scorer=data/evaluation/IWSLT2022/scorer.py

f_r=$TMPDIR/formal_results
i_r=$TMPDIR/informal_results
rm -f $f_r $i_r

for index in "${!formal_prompts[@]}"; do
        f_p="${formal_prompts[$index]}"
        i_p="${informal_prompts[$index]}"

        head -n 600 $TMPDIR/tmp.hyp > $TMPDIR/test.en-$lang.formal.hyp
        sed -i '1,600d' $TMPDIR/tmp.hyp

  printf "Calculating scores for FORMAL prompt (formal accuracy matters)\nPrompt is %s\n" "$f_p"
  python $scorer -hyp $TMPDIR/test.en-$lang.formal.hyp \
  -f $data/formality-control.test.en-$lang.formal.annotated.$lang \
  -if $data/formality-control.test.en-$lang.informal.annotated.$lang >> $f_r
  tail -n1 $f_r

  head -n 600 $TMPDIR/tmp.hyp > $TMPDIR/test.en-$lang.informal.hyp
  sed -i '1,600d' $TMPDIR/tmp.hyp

  printf "\n\n\nCalculating scores for INFORMAL prompt (informal accuracy matters)\nPrompt is %s\n" "$i_p"
  python $scorer -hyp $TMPDIR/test.en-$lang.informal.hyp \
  -f $data/formality-control.test.en-$lang.formal.annotated.$lang \
  -if $data/formality-control.test.en-$lang.informal.annotated.$lang >> $i_r
  tail -n1 $i_r
  printf "\n\n\n\n"
done
# Get prompt-wide results
grep Formal $f_r | awk -F' |,' '{ sum += $3; n++ } END { print "Formal Acc: " sum / n }' > $TMPDIR/acc
grep Formal $f_r | awk -F' |,' '{ if ($3 > max) max = $3 } END { print "Best Formal Acc: " max }' >> $TMPDIR/acc
grep Formal $i_r | awk -F' |,' '{ sum += $7; n++ } END { print "Informal Acc: " sum / n }' >> $TMPDIR/acc
grep Formal $i_r | awk -F' |,' '{ if ($7 > max) max = $7 } END { print "Best Informal Acc: " max }' >> $TMPDIR/acc

cat $TMPDIR/acc
