#!/bin/bash

fourier meta-modify "path/to/fourierAnalogue.frad" \
\
--metadata "Metadata Title" "Metadata contents" \
--jsonmeta "path/to/metadata.json" \
--image "path/to/image/file"

# JSON Metadata goes prior comparing Metadata option.