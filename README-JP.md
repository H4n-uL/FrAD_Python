# Fourier Analogue-in-Digital

## プロジェクト概要

[Project Archivist](https://mikhael-openworkspace.notion.site/Project-Archivist-e512fa7a21474ef6bdbd615a424293cf)の一環として、アナログ信号をそのままデジタルの中に収めるという目標を持って開発されました。 詳細は[Notion](https://mikhael-openworkspace.notion.site/Fourier-Analogue-in-Digital-d170c1760cbf4bb4aaea9b1f09b7fead?pvs=4)で確認することができます。

**警告： Fourier Analogue-in-Digital Compactは、そのフォーマットが絶えず変更されており、不安定です。まだFourier Analogue-in-Digitalをエンコードする際には、損失圧縮オプションを使用しないでください。**

## インストール方法

1. Git zipをダウンロード
2. Python(3.11^)をインストール
3. install.sh を実行
4. source ~/.*shrc でシェルを再実行

インストールはまだUnix系OSだけサポートします。

## 外部リソース

[Python](https://github.com/python/cpython), [FFmpeg](https://github.com/FFmpeg/FFmpeg), [QAAC](https://github.com/nu774/qaac), [QTFiles](https://github.com/AnimMouse/QTFiles), afconvert

### pipパッケージ

1. mdctn
2. numpy
3. reedsolo
4. scipy
5. sounddevice

## 使用方法

エンコーディング

```bash
fourier encode "path/to/audio.flac" \
--bits 32 \                                        # ビット深度
\  # オプション
--output "path/to/fourierAnalogue.frad" \          # 出力ファイル
--frame_size 2048 \                                # ブロックあたりのサンプル数
--enable_ecc \                                     # ECC使用可否
--data_ecc_size 128 20 \                           # データブロックとECCブロックのサイズ
--little_endian \                                  # エンディアン
--gain -6 \                                        # ゲイン
--dbfs \                                           # ゲイン単位フラグ
--metadata "Metadata Title" "Metadata contents" \  # メタデータ
--jsonmeta "path/to/metadata.json" \               # メタデータ json, --metadata より優先されます。
--image "path/to/image/file" \                     # 画像ファイル
--verbose
```

デコーディング

``` bash
fourier decode "path/to/fourierAnalogue.frad" \
\  # オプション
--bits 32 \                      # 無損失圧縮コーデックのビット深度(8, 16, 32をサポートします)
--enable_ecc \                   # ECC使用可否
--gain 1.2 \                     # ゲイン
--dbfs \                         # ゲイン単位フラグ
--output "path/for/audio.aac" \  # 出力ファイル
--codec "codec" \                # コーデックの種類
--quality "320000c" \            # 損失圧縮コーデックの品質 (例は固定320 kbps)
--verbose
```

プレー

``` bash
fourier play "path/to/fourierAnalogue.frad" \
\  # オプション
--key keys \     # プレーキー
--speed speed \  # プレー速度
--gain 3.4 \     # ゲイン
--dbfs \         # ゲイン単位フラグ
--enable_ecc \   # ECC使用可否
--verbose
```

メタデータ編集

``` bash
fourier meta-modify "path/to/fourierAnalogue.frad" \
\  # オプション
--metadata "Metadata Title" "Metadata contents" \  # メタデータ
--jsonmeta "path/to/metadata.json" \               # メタデータ json, --metadata より優先されます。
--image "path/to/image/file" \                     # 画像ファイル
```

メタデータ抽出

``` bash
fourier parse "path/to/fourierAnalogue.frad" \
\  # オプション
--output "path/for/metadata" \  # Output file.meta.json, Output file.meta.image
```

ECCパッキング/リパッキング

```bash
fourier ecc "path/to/fourierAnalogue.frad" \
\  # オプション
--verbose
```

ソフトウェアレコーディング

```bash
fourier encode "path/to/fourierAnalogue.frad" \
\  # オプション
--bits 32 \                                        # ビット深度
--frame_size 2048 \                                # ブロックあたりのサンプル数
--enable_ecc \                                     # ECC使用可否
--data_ecc_size 128 20 \                           # データブロックとECCブロックのサイズ
--little_endian \                                  # エンディアン
--metadata "Metadata Title" "Metadata contents" \  # メタデータ
--jsonmeta "path/to/metadata.json" \               # メタデータ json, --metadata より優先されます。
--image "path/to/image/file" \                     # 画像ファイル
```

メタデータJSON

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

## 寄付方法

リポジトリで新しいブランチを作成し、修正して、私にMergeリクエストで審査を受けてください。実はこいつはザコなので、大抵は全部通ります。

## ライセンス

### 適用対象

このライセンスは、Fourier Analogue-in-Digitalの著作者人格権、知的財産権、Gitリポジトリ、ソースコードなど全体に適用されます。

### 許可

すべての人は、ソースコードの使用と再加工、再配布、改善及び貢献、プロジェクトの実装方法の変更(ただし、このソースコードと互換性がなければなりません。)、商業的なプログラムの一部として活用することができます。

### 制限

このリポジトリを単独で商業的に使用することは禁止されています。

プロジェクトの原著作者は明示する必要はありません。ただし、貢献をしていない第三者が自分がこのプロジェクトの著作権を主張することは禁止されています。貢献をした場合、原著者と貢献者の名前を併記することができます。

このプロジェクトの核心概念で特許権や商標権を主張することはできません。 ただし、付加機能を開発して適用した場合、その付加機能に限り、独立した特許権や商標権を主張することができます。

DRMの適用は固く禁じられています。万人に開かれたフォーマットであるFourier Analogue-in-DigitalにDRMを適用することは容認できません。

### 免責事項

このソースコードを使用することによって発生するいかなる結果についても、原著作者と貢献者は責任を負いません。 実際、責任を負いたくても負えません。

## 開発者情報

ハンウル, <jun061119@proton.me>
