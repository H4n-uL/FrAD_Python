#!/bin/bash

python3 Python/main.py play "path/to/fourierAnalogue.frad" \
\
--key keys \
--speed speed \
--enable_ecc \
--verbose

# Keys and Speed value cannot be applied in a command at same time