#!/bin/sh
for keyword in Avg Sum Min Max Gini "20:20"
do
  fname=$(echo $1 | cut -d '.' -f 1)
  fname=$fname"_"$keyword".txt"
  echo $fname
  cat $1 | grep $keyword > $fname 
done
