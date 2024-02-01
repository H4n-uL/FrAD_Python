#!/bin/bash
if ! command -v python3 &> /dev/null
then
    echo "Python 3 not found, Please install Python 3 before installing."
    exit 1
fi

if [ -d "~/FrAD-Codec/" ]; then
    read -p "Do you want to delete existing files? (Y/N): " DELETE_FILES
    if [ "$DELETE_FILES" = "Y" ] || [ "$DELETE_FILES" = "y" ]; then
        rm -r ~/FrAD-Codec/
    else
        echo "Installation aborted."
        exit 1
    fi
fi

# 파일 복사
cp -r src/ ~/FrAD-Codec/

# 필요한 패키지 설치
python3 -m pip install -r ~/FrAD-Codec/requirements.txt

# alias 설정

for file in ~/.*shrc
do
    if [ -w "$file" ]
    then
        sed -i '' '/alias fourier=/d' "$file"
        echo 'alias fourier="python3 ~/FrAD-Codec/main.py"' >> "$file"
    fi
done

echo "Installation completed. Please restart your shell to apply the changes."