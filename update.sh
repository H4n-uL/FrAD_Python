#!/bin/bash
set -e

if ! command -v python3 &> /dev/null
then
    echo "Python3 not installed, Please install Python3."
    exit
fi

if [ "$FOURIER_PATH/FrAD-Codec/main.py" ]; then
    rsync -a --exclude='__pycache__' --exclude='.DS_Store' "$(dirname "$0")/src/" $FOURIER_PATH/FrAD-Codec || { echo "Failed copying files."; exit 1; }
    python3 -m pip install -r $FOURIER_PATH/FrAD-Codec/requirements.txt -q  || { echo "Failed installing pip packages."; exit 1; }
    echo "Update completed."; exit 0;
else
    ./install.sh; exit 0;
fi
