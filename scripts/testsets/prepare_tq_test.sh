#!/bin/bash
# Prepares the ZOO and OpenSubtitles data for evaluation.

set -e

spm_encode=fairseq/scripts/spm_encode.py

# Prepare EAMT test set
<<<<<<< HEAD
data=data/en-pl/eamt22_test
data_bin=fairseq/data-bin/test.eamt22.en.pl
spm_model=training_data/baseline.en.pl/spm.bpe.model

echo "encoding dev with learned BPE..."
python ${spm_encode} \
    --model $spm_model \
    --output_format=piece \
    --inputs $data/eamt22.test.en $data/eamt22.test.pl \
    --outputs $data/eamt22.test.bpe.en $data/eamt22.test.bpe.pl

# Binarise
fairseq-preprocess --joined-dictionary \
    --source-lang en --target-lang pl \
    --tgtdict fairseq/data-bin/baseline.en.pl/dict.pl.txt \
    --testpref $data/eamt22.test.bpe \
    --destdir ${data_bin} \
    --workers 20

for embedder in distilbert minilm mpnet hash; do
	python context_embeddings/small_embeddings.py --path $data --dest $data --model $embedder --lang pl
	cp $data/test.$embedder.cxt.bin $data_bin/test.en-pl.$embedder.cxt.bin
	cp $data/test.$embedder.cxt.idx $data_bin/test.en-pl.$embedder.cxt.idx
	cp $data/test.$embedder.json $data_bin/test.en-pl.$embedder.json
done

exit 1

src=en
for tgt in de pl fr; do
  data=data/$src-$tgt
  data_bin=fairseq/data-bin/test.$src.$tgt
  data_bin_reverse=fairseq/data-bin/test.$tgt.$src
  

  # Encode test data with sentencepiece
  spm_model=training_data/baseline.$src.$tgt/spm.bpe.model

  echo "encoding dev with learned BPE..."
  python ${spm_encode} \
      --model $spm_model \
      --output_format=piece \
      --inputs $data/test.$src $data/test.$tgt \
      --outputs $data/test.bpe.$src $data/test.bpe.$tgt

  # Binarise forward direction
  fairseq-preprocess --joined-dictionary \
      --source-lang ${src} --target-lang ${tgt} \
      --tgtdict fairseq/data-bin/baseline.${src}.${tgt}/dict.${tgt}.txt \
      --testpref $data/test.bpe \
      --destdir ${data_bin} \
      --workers 20
  
  # Preprocess backward direction (tgt->src)
  spm_model=training_data/baseline.$tgt.$src/spm.bpe.model

  echo "encoding dev with learned BPE..."
  python $spm_encode \
      --model $spm_model \
      --output_format=piece \
      --inputs $data/test.$tgt $data/test.$src \
      --outputs $data/test.bpe.$tgt $data/test.bpe.$src

  fairseq-preprocess --joined-dictionary \
      --source-lang $tgt --target-lang $src \
      --tgtdict fairseq/data-bin/baseline.$tgt.$src/dict.$src.txt \
      --testpref $data/test.bpe \
      --destdir $data_bin_reverse \
      --workers 20
 # Remove BPE files as they need to be generated anew each time
  rm $data/test.bpe.$src $data/test.bpe.$tgt

  # Copy cxt embeddings to data_bin
  for embedder in mpnet minilm distilbert; do
    #bash context_embeddings/small_on_scratch.sh $data $src $embedder
    for content in $src.bin $src.idx cxt.bin cxt.idx json; do
      cp $data/dynamic/test.$embedder.$content $data_bin/test.$src-$tgt.$embedder.$content
    done
    # multilingual for reverse direction
    e="$embedder"-multilingual
    #bash context_embeddings/small_on_scratch.sh $data $tgt $e
    for content in $tgt.bin $tgt.idx cxt.bin cxt.idx json; do
      cp $data/dynamic/test.$e.$content $data_bin_reverse/test.$tgt-$src.$e.$content
    done
  done
done

