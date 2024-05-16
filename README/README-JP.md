# Fourier Analogue-in-Digital

## プロジェクト概要

[Project Archivist](https://mikhael-openworkspace.notion.site/Project-Archivist-e512fa7a21474ef6bdbd615a424293cf)の一環として、アナログ信号をそのままデジタルの中に収めるという目標を持って開発されました。 詳細は[Notion](https://mikhael-openworkspace.notion.site/Fourier-Analogue-in-Digital-d170c1760cbf4bb4aaea9b1f09b7fead?pvs=4)で確認することができます。

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

## メタデータJSON例

```json
[
    {"key": "KEY",                          "type": "string", "value": "VALUE"},
    {"key": "AUTHOR",                       "type": "string", "value": "ハンウル"},
    {"key": "キーとStringタイプのエンコーディング", "type": "string", "value": "UTF-8"},
    {"key": "Base64 サポート",                "type": "base64", "value": "QmFzZTY044Gu5L6L"},
    {"key": "ファイルサポート",                 "type": "base64", "value": "5pyA5aSnMjU2IFRpQuOBvuOBp+OCteODneODvOODiA=="},
    {"key": "未対応文字なし",                  "type": "string", "value": "Unicodeにあるどの文字でも互換性があります！"},
    {"key": "重複キーサポート",                 "type": "string", "value": "キーが重複するようにすると？"},
    {"key": "重複キーサポート",                 "type": "string", "value": "パンパカパーン！"}
]
```

## 寄付方法

リポジトリで新しいブランチを作成し、修正して、私にMergeリクエストで審査を受けてください。実はこいつはザコなので、大抵は全部通ります。

## 開発者情報

ハンウル, <jun061119@proton.me>
