#!/bin/bash

python3 Python/main.py decode "path/to/fourierAnalogue.fra" \
\
--bits 32 \
--output "path/for/audio.aac" \
--codec "codec" \
--quality "320000c"