# Fourier Analogue-in-Digital

## Project Overview

It was developed as part of Project Archivist with the goal of keeping analogue signals in digital. More information can be found in the [Notion](https://mikhael-openworkspace.notion.site/Project-Archivist-e512fa7a21474ef6bdbd615a424293cf).

Note: The purpose of an archive is always to make it easy for you to access important data. Please be sure to back up any important audio files.

## How to install

1. Download the Git zip
2. Install Python (3.11^)
3. Execute install.sh
4. Restart shell with source ~/.*shrc

Installation is still only supported on Unix-like OS.

## How to use

Encoding

```bash
fourier encode "path/to/audio.flac" \
--bits 32 \                                        # Bit depth
\  # Optional
--output "path/to/fourierAnalogue.frad" \          # Output file
--samples_per_block 2048 \                         # Samples per block
--enable_ecc \                                     # ECC enabled or not
--data_ecc_size 128 20 \                           # Sizes of data block and ECC block when ECC enabled
--metadata "Metadata Title" "Metadata contents" \  # Metadata
--jsonmeta "path/to/metadata.json" \               # Metadata json, will override --metadata.
--image "path/to/image/file" \                     # Image file
--verbose
```

Decoding

```bash
fourier decode "path/to/fourierAnalogue.frad" \
\  # Optional
--bits 32 \                      # Bit depth for lossless compression codecs (supports 8, 16, 32)
--enable_ecc \                   # ECC verification or not
--output "path/for/audio.aac" \  # Output file
--codec "codec" \                # Codec type
--quality "320000c" \            # Quailty factor for lossy compression codecs (example is constant 320 kbps)
--verbose
```

Playback

```bash
fourier play "path/to/fourierAnalogue.frad" \
\  # Optional
--key keys \     # Playback keys
--speed speed \  # Playback speed
--enable_ecc \   # ECC verification or not
--verbose
```

Edit metadata

```bash
fourier meta-modify "path/to/fourierAnalogue.frad" \
\  # Optional
--metadata "Metadata Title" "Metadata contents" \  # Metadata
--jsonmeta "path/to/metadata.json" \               # Metadata json, will override --metadata.
--image "path/to/image/file" \                     # Image file
```

Extract metadata

```bash
fourier parse "path/to/fourierAnalogue.frad"
```

ECC packing/repacking

```bash
fourier ecc "path/to/fourierAnalogue.frad" \
\  # Optional
--verbose
```

Example of .json file is in Example folder.

## How to contribute

![Don?know](https://item.kakaocdn.net/do/4a675e36e71c3538c5e7ada87a2b28fef43ad912ad8dd55b04db6a64cddaf76d)

## Licence

### Coverage

This licence applies to the entirety of the Moral rights, Intellectual property rights, Git repository, source code, etc. of Fourier Analogue-in-Digital.

### Permissions

Everyone is permitted to use, reproduce, redistribute, improve and contribute to the source code, modify the project implementation (which must be compatible with this source code), and utilise it as part of a commercial program.

### Restrictions

Solo commercial use of any implementation of this project is prohibited.

The original authors of the project do not need to be acknowledged. However, third parties who have not contributed are prohibited from claiming copyright in this project. If a contribution is made, the contributor's name can be included next to the original author's name.

The core concepts of this project may not be patented or trademarked. However, if additional features are developed and applied, independent patent or trademark rights may be claimed for those features only.

### Disclaimer

The original authors and contributors are not responsible for any consequences arising from the use of this source code; in fact, they cannot be held liable even if they wanted to be.

## Developer information

HaמuL, <jun061119@proton.me>
