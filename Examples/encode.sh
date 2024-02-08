#!/bin/bash

fourier encode "path/to/audio.flac" \
--bits 32 \
\
--output "path/for/fourierAnalogue.frad" \
--samples_per_block 2048 \
--enable_ecc \
--data_ecc_size 128 20 \
--metadata "Metadata Title" "Metadata contents" \
--jsonmeta "path/to/metadata.json" \
--image "path/to/image/file" \
--verbose

# -jm goes prior comparing -m option.