#!/bin/bash
for tgt in de fr pl; do
	data=data/en-$tgt/dynamic
	data_bin=fairseq/data-bin/mtcue.en.$tgt.next_os
#	for embedder in hash; do
#		for c in cxt.bin cxt.idx en.bin en.idx json; do
#			#if [ ! -f $data_bin/train.en-$tgt.$embedder.$c ]; then
#			        cp $data/train.$embedder.$c $data_bin/train.en-$tgt.$embedder.$c
#			        cp $data/dev.$embedder.$c $data_bin/valid.en-$tgt.$embedder.$c
#			#fi
#		done
#	done
	data_bin=fairseq/data-bin/mtcue.$tgt.en.next_os
	for embedder in hash; do
		for c in cxt.bin cxt.idx $tgt.bin $tgt.idx json; do
#			if [ ! -f $data_bin/train."$tgt"-en.$embedder.$c ]; then
			        cp $data/dev."$embedder"-multilingual.$c $data_bin/valid."$tgt"-en.$embedder.$c
			        cp $data/train."$embedder"-multilingual.$c $data_bin/train."$tgt"-en.$embedder.$c
#			fi
		done
	done
done
