#!/bin/bash

python3 Python/main.py decode "path/to/audio.flac" \
\
-b 32 \
-o "path/for/fourierAnalogue.fra" \
-c "codec" \
-q "320k"