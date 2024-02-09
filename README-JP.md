# Fourier Analogue-in-Digital

## プロジェクト概要

Project Archivistの一環として、アナログ信号をそのままデジタルの中に収めるという目標を持って開発されました。 詳細は[Notion](https://mikhael-openworkspace.notion.site/Project-Archivist-e512fa7a21474ef6bdbd615a424293cf)で確認することができます。

注意: アーカイブの目的は、いつでも簡単に重要なデータにアクセスできるようにすることです。重要な音源ファイルは必ずバックアップしておいてください。

## インストール方法

1. Git zipをダウンロード
2. Python(3.11^)をインストール
3. install.sh を実行
4. source ~/.*shrc でシェルを再実行

インストールはまだUnix系OSだけサポートします。

## 使用方法

エンコーディング

```bash
fourier encode "path/to/audio.flac" \
--bits 32 \                                        # ビット深度
\  # オプション
--output "path/to/fourierAnalogue.frad" \          # 出力ファイル
--samples_per_block 2048 \                         # ブロックあたりのサンプル数
--enable_ecc \                                     # ECC使用可否
--data_ecc_size 128 20 \                           # データブロックとECCブロックのサイズ
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
fourier parse "path/to/fourierAnalogue.frad"
```

ECCパッキング/リパッキング

```bash
fourier ecc "path/to/fourierAnalogue.frad" \
\  # オプション
--verbose
```

.jsonファイルの例はExamplesフォルダにあります。

## 寄付方法

![わかん?にゃい](https://item.kakaocdn.net/do/4a675e36e71c3538c5e7ada87a2b28fef43ad912ad8dd55b04db6a64cddaf76d)

## ライセンス

### 適用対象

このライセンスは、Fourier Analogue-in-Digitalの著作者人格権、知的財産権、Gitリポジトリ、ソースコードなど全体に適用されます。

### 許可

すべての人は、ソースコードの使用と再加工、再配布、改善及び貢献、プロジェクトの実装方法の変更(ただし、このソースコードと互換性がなければなりません。)、商業的なプログラムの一部として活用することができます。

### 制限

このプロジェクトの実装体を単独で商業的に使用することは禁止されています。

プロジェクトの原著作者は明示する必要はありません。ただし、貢献をしていない第三者が自分がこのプロジェクトの著作権を主張することは禁止されています。貢献をした場合、原著者と貢献者の名前を併記することができます。

このプロジェクトの核心概念で特許権や商標権を主張することはできません。 ただし、付加機能を開発して適用した場合、その付加機能に限り、独立した特許権や商標権を主張することができます。

### 免責事項

このソースコードを使用することによって発生するいかなる結果についても、原著作者と貢献者は責任を負いません。 実際、責任を負いたくても負えません。

## 開発者情報

ハンウル, <jun061119@proton.me>
