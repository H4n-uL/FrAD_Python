#!/bin/bash

fourier decode "path/to/fourierAnalogue.frad" \
\
--bits 32 \
--enable_ecc \
--output "path/for/audio.aac" \
--codec "codec" \
--quality "320000c" \
--verbose