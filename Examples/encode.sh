#!/bin/bash

python3 Python/fourierAnalogue.py encode \
\
-i "path/to/audio.flac" \
-b 32 \
\
-o "path/for/fourierAnalogue.fra" \
-n 48000 \
-e \
-m "Metadata Title" "Metadata contents" \
-jm "path/to/metadata.json" \
-img "path/to/image/file"

# -jm goes prior comparing -m option.