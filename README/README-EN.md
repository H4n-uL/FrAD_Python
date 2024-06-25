# Fourier Analogue-in-Digital

## Project Overview

Python implementation of [AAPM](https://mikhael-openworkspace.notion.site/Project-Archivist-e512fa7a21474ef6bdbd615a424293cf)@Audio-8151. More information can be found in the [Notion](https://mikhael-openworkspace.notion.site/Fourier-Analogue-in-Digital-d170c1760cbf4bb4aaea9b1f09b7fead?pvs=4).

## How to install

1. Download the Git zip
2. Install Python (3.11^)
3. Execute install.sh
4. Restart shell with source ~/.*shrc

Installation is still only supported on Unix-like OS.

## External Resources

[Python](https://github.com/python/cpython), [FFmpeg](https://github.com/FFmpeg/FFmpeg), [QAAC](https://github.com/nu774/qaac), [QTFiles](https://github.com/AnimMouse/QTFiles), afconvert

### pip Packages

1. numpy
2. scipy
3. reedsolo
4. sounddevice

## Metadata JSON example

Metadata JSON

```json
[
    {"key": "KEY",                              "type": "string", "value": "VALUE"},
    {"key": "AUTHOR",                           "type": "string", "value": "H4n_uL"},
    {"key": "Key & String value encoding type", "type": "string", "value": "UTF-8"},
    {"key": "Base64 type Value",                "type": "base64", "value": "QmFzZTY0IEV4YW1wbGU="},
    {"key": "File is also available",           "type": "base64", "value": "U3VwcG9ydHMgdXAgdG8gMjU2IFRpQg=="},
    {"key": "No unsupported characters",        "type": "string", "value": "All utf-8/base64 metadata is allowed!"},
    {"key": "Supports duplicate keys",          "type": "string", "value": "See what happens!"},
    {"key": "Supports duplicate keys",          "type": "string", "value": "Voilà!"}
]
```

## How to contribute

Simply create a new branch in the repository, make your changes, and submit a merge request to me for approval. Pretty much anything will pass if it conforms to the FrAD format standard.

## Implementation requirements

1. Essential

    ```markdown
    FrAD/
        fourier
        profiles/
            profile1
            tools/
                p1tools
        decoder
        encoder
        header
        common
        tools/
            headb
            ecc
        repack
    ```

2. Optional

    ```markdown
    main
    FrAD/
        player
        record
        tools/
            update
            argparse
        res/
            AppleAAC.Win.tar.gz -> AppleAAC
    ...and all the other stuff
    ```

## Developer information

HaמuL, <jun061119@proton.me>
