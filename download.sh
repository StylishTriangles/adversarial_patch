#!/bin/bash
# usage:	./download.sh links_to_download.txt path/to/data/directory
# example:	./download.sh datasources/cat.txt training/cat


mkdir -p $2

for link in $(cat $1)
do
	$(cd $2; wget -t 1 -T 10 $link)
done
