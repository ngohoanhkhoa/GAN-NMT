#!/bin/bash

PREFIX="https://raw.githubusercontent.com/cmu-mtlab/meteor/master/data/paraphrase"
SAVEDIR="nmtpy/external/data"

for lang in cz de en es fr ru; do
  if [ ! -f "${SAVEDIR}/paraphrase-${lang}.gz" ]; then
    echo "Downloading $lang paraphrase data..."
    curl "${PREFIX}-${lang}.gz" -o "${SAVEDIR}/paraphrase-${lang}.gz"
  fi
done
