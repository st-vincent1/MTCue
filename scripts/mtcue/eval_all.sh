#!/bin/bash

for arch in merger aug_parallel aug_sequential concat; do
	for lang in de fr pl; do
		echo $arch $lang
		bash scripts/mtcue/evaluate.sh --src en --tgt $lang --suffix next_os --arch mtcue_"$arch"_qk --context-model minilm
		bash scripts/mtcue/evaluate.sh --src $lang --tgt en --suffix next_os --arch mtcue_"$arch"_qk --context-model minilm
	done
done	
