#!/bin/bash

for tgt in pl; do
	data=data/en-$tgt/eamt22_test/dynamic
	data_bin=fairseq/data-bin/test.eamt22.en.$tgt
	for embedder in minilm hash; do
		for c in cxt.bin cxt.idx en.bin en.idx json; do
		        cp $data/test.$embedder.$c $data_bin/test.en-$tgt.$embedder.$c
		done
	done
done

exit 1
for tgt in de fr pl; do
	data=data/en-$tgt/dynamic
	data_bin=fairseq/data-bin/test.en.$tgt
	for embedder in distilbert minilm mpnet hash; do
		for c in cxt.bin cxt.idx en.bin en.idx json; do
		        cp $data/test.$embedder.$c $data_bin/test.en-$tgt.$embedder.$c
		done
	done
	data_bin=fairseq/data-bin/test.$tgt.en
	for embedder in distilbert minilm mpnet hash; do
		for c in cxt.bin cxt.idx $tgt.bin $tgt.idx json; do
		        cp $data/test."$embedder"-multilingual.$c $data_bin/test.$tgt-en.$embedder.$c
		done
	done
done

