#!/bin/bash
set -e

if ! command -v python3 &> /dev/null
then
    echo "Python3 not installed, Please install Python3."
    exit
fi

if [ -d "$FOURIER_PATH/FrAD-Codec" ]; then
    read -p "Fourier Analogue-in-Digital is already installed. Would you like to remove and reinstall? (Y/N) " yn
    case $yn in
        [Yy]* ) rm -r $FOURIER_PATH/FrAD-Codec;;
        * ) echo "Installation Aborted."; exit 1;;
    esac
fi

read -p "Installation path(Default is $HOME): " FOURIER_PATH
if [ -z "$FOURIER_PATH" ]; then
    FOURIER_PATH=$HOME
fi
export FOURIER_PATH

mkdir -p $FOURIER_PATH/FrAD-Codec || { exit 1; }
rsync -a --exclude='__pycache__' --exclude='.DS_Store' "$(dirname "$0")/src/" $FOURIER_PATH/FrAD-Codec || { echo "Failed copying files."; exit 1; }

for rcfile in `ls -a ~ | grep 'shrc$'`; do
    if grep -q 'FOURIER_PATH' ~/$rcfile; then
        sed -i '' "/FOURIER_PATH/d" ~/$rcfile
    fi
    if grep -q 'alias fourier=' ~/$rcfile; then
        sed -i '' "/alias fourier=/d" ~/$rcfile
    fi
    echo "export FOURIER_PATH=$FOURIER_PATH" >> ~/$rcfile
    echo "alias fourier='python3 \$FOURIER_PATH/FrAD-Codec/main.py'" >> ~/$rcfile
done

python3 -m pip install -r $FOURIER_PATH/FrAD-Codec/requirements.txt -q  || { echo "Failed installing pip packages."; exit 1; }

echo "Installation completed. Please restart your shell to apply changes."
