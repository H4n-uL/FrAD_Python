#!/bin/bash

python3 Python/main.py encode "path/to/audio.flac" \
--bits 32 \
\
-output "path/for/fourierAnalogue.fra" \
--nsr 48000 \
-e \
-m "Metadata Title" "Metadata contents" \
-jm "path/to/metadata.json" \
-img "path/to/image/file"

# -jm goes prior comparing -m option.