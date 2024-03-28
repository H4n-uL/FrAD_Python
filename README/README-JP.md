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

1. numpy
2. scipy
3. reedsolo
4. sounddevice

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
    {"key": "KEY",                          "type": "string", "value": "VALUE"},
    {"key": "AUTHOR",                       "type": "string", "value": "H4n_uL"},
    {"key": "キーとStringタイプのエンコーディング", "type": "string", "value": "UTF-8"},
    {"key": "Base64 サポート",                "type": "base64", "value": "QmFzZTY044Gu5L6L"},
    {"key": "ファイルサポート",                 "type": "base64", "value": "5pyA5aSnMjU2IFRpQuOBvuOBp+OCteODneODvOODiA=="},
    {"key": "未対応文字なし",                  "type": "string", "value": "Unicodeにあるどの文字でも互換性があります！"},
    {"key": "重複キーサポート",                 "type": "string", "value": "キーが重複するようにすると？"},
    {"key": "重複キーサポート",                 "type": "string", "value": "ほら！"}
]
```

## 寄付方法

リポジトリで新しいブランチを作成し、修正して、私にMergeリクエストで審査を受けてください。実はこいつはザコなので、大抵は全部通ります。

## 開発者情報

ハンウル, <jun061119@proton.me>
