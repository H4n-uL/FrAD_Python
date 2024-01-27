#!/bin/bash

python3 Python/main.py encode "path/to/audio.flac" \
--bits 32 \
\
--output "path/for/fourierAnalogue.fra" \
--enable_ecc \
--metadata "Metadata Title" "Metadata contents" \
--jsonmeta "path/to/metadata.json" \
--image "path/to/image/file" \
--verbose

# -jm goes prior comparing -m option.