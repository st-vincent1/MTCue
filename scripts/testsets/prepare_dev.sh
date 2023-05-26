#!/bin/bash
# Prepares the ZOO and OpenSubtitles data for evaluation.

set -e

spm_encode=fairseq/scripts/spm_encode.py

src=en
for tgt in de fr pl; do
  data=data/$src-$tgt
  data_bin=fairseq/data-bin/valid.$src.$tgt
  data_bin_reverse=fairseq/data-bin/valid.$tgt.$src
  

  # Encode valid data with sentencepiece
  spm_model=training_data/baseline.$src.$tgt/spm.bpe.model

  echo "encoding dev with learned BPE..."
  python ${spm_encode} \
      --model $spm_model \
      --output_format=piece \
      --inputs $data/dev.$src $data/dev.$tgt \
      --outputs $data/dev.bpe.$src $data/dev.bpe.$tgt

  # Binarise forward direction
  fairseq-preprocess --joined-dictionary \
      --source-lang ${src} --target-lang ${tgt} \
      --tgtdict fairseq/data-bin/baseline.${src}.${tgt}/dict.${tgt}.txt \
      --testpref $data/dev.bpe \
      --destdir ${data_bin} \
      --workers 20
  
  # Preprocess backward direction (tgt->src)
  spm_model=training_data/baseline.$tgt.$src/spm.bpe.model

  echo "encoding dev with learned BPE..."
  python $spm_encode \
      --model $spm_model \
      --output_format=piece \
      --inputs $data/dev.$tgt $data/dev.$src \
      --outputs $data/dev.bpe.$tgt $data/dev.bpe.$src

  fairseq-preprocess --joined-dictionary \
      --source-lang $tgt --target-lang $src \
      --tgtdict fairseq/data-bin/baseline.$tgt.$src/dict.$src.txt \
      --testpref $data/dev.bpe \
      --destdir $data_bin_reverse \
      --workers 20
  # Remove BPE files as they need to be generated anew each time
  rm $data/dev.bpe.$src $data/dev.bpe.$tgt

  # Copy cxt embeddings to data_bin
  for embedder in sentence-transformers multilingual distilbert; do
    for content in cxt.bin cxt.json; do
      cp $data/dev.$embedder.$content $data_bin/valid.$src-$tgt.$embedder.$content
      cp $data/dev.$embedder.$content $data_bin_reverse/valid.$tgt-$src.$embedder.$content
    done
  done
done

