#!/bin/bash
# Usage:	./splitter.sh training/data/directory validation/data/directory
# Example:	./splitter.sh training validation
#	Assuming all your images are categorized in training directory
# IMPORTANT: training directory must contain only directories
# The split between training and validation is roughly 25000 to 7000

source=$1
dest=$2

for dir in $(ls $source)
do
	for file in $(ls $source/$dir)
	do
		if [ $RANDOM -ge 25000 ]; then
			mkdir -p $dest/$dir
			mv $source/$dir/$file $dest/$dir/$file
		fi
	done
done
