#!/bin/bash

python3 Python/main.py modify \
\
-i "path/to/audio.flac" \
\
-m "Metadata Title" "Metadata contents" \
-jm "path/to/metadata.json" \
-img "path/to/image/file"

# -jm goes prior comparing -m option.