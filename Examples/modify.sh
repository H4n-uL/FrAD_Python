#!/bin/bash

python3 Python/main.py meta-modify "path/to/fourierAnalogue.frad" \
\
--meta "Metadata Title" "Metadata contents" \
--jsonmeta "path/to/metadata.json" \
--image "path/to/image/file"

# JSON Metadata goes prior comparing Metadata option.