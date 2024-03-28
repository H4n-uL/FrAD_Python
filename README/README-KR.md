# Fourier Analogue-in-Digital

## 프로젝트 개요

[Project Archivist](https://mikhael-openworkspace.notion.site/Project-Archivist-e512fa7a21474ef6bdbd615a424293cf)의 일환으로 아날로그 신호를 그대로 디지털 속에 담아낸다는 목표를 가지고 개발되었습니다. 자세한 내용은 [Notion](https://mikhael-openworkspace.notion.site/Fourier-Analogue-in-Digital-d170c1760cbf4bb4aaea9b1f09b7fead?pvs=4)에서 확인하실 수 있습니다.

**경고: Fourier Analogue-in-Digital Compact는 지속적으로 그 포맷이 변경되고 있어 불안정합니다. 부디 아직은 Fourier Analogue-in-Digital을 인코딩할 때 손실 압축 옵션을 사용하지 말아 주시기 바랍니다.**

## 설치 방법

1. Git zip 다운로드
2. Python(3.11^) 설치
3. install.sh 실행
4. source ~/.*shrc로 shell 재실행

설치는 아직 유닉스 계열 OS에서만 지원합니다.

## 외부 리소스

[Python](https://github.com/python/cpython), [FFmpeg](https://github.com/FFmpeg/FFmpeg), [QAAC](https://github.com/nu774/qaac), [QTFiles](https://github.com/AnimMouse/QTFiles), afconvert

### pip 패키지

1. numpy
2. scipy
3. reedsolo
4. sounddevice

## 사용 방법

인코딩

```bash
fourier encode "path/to/audio.flac" \
--bits 32 \                                        # 비트 심도
\  # 선택 사항
--output "path/to/fourierAnalogue.frad" \          # 출력 파일
--frame_size 2048 \                                # 블록당 샘플 수
--enable_ecc \                                     # ECC 활성화 여부
--data_ecc_size 128 20 \                           # 데이터 블록과 ECC 블록의 크기
--little_endian \                                  # 엔디언
--gain -6 \                                        # 게인
--dbfs \                                           # 게인 단위 플래그
--metadata "Metadata Title" "Metadata contents" \  # 메타데이터
--jsonmeta "path/to/metadata.json" \               # 메타데이터 json, --metadata보다 우선시됩니다.
--image "path/to/image/file" \                     # 이미지 파일
--verbose
```

디코딩

```bash
fourier decode "path/to/fourierAnalogue.frad" \
\  # 선택 사항
--bits 32 \                      # 무손실 압축 코덱의 비트 심도 (8, 16, 32를 지원합니다.)
--enable_ecc \                   # ECC 확인 여부
--gain 1.2 \                     # 게인
--dbfs \                         # 게인 단위 플래그
--output "path/for/audio.aac" \  # 출력 파일
--codec "codec" \                # 코덱 종류
--quality "320000c" \            # 손실 압축 코덱의 품질 (예시는 고정 320 kbps)
--verbose
```

재생

```bash
fourier play "path/to/fourierAnalogue.frad" \
\  # 선택 사항
--key keys \     # 재생 키
--speed speed \  # 재생 속도
--gain 3.4 \     # 게인
--dbfs \         # 게인 단위 플래그
--enable_ecc \   # ECC 확인 여부
--verbose
```

메타데이터 편집

```bash
fourier meta-modify "path/to/fourierAnalogue.frad" \
\  # 선택 사항
--metadata "Metadata Title" "Metadata contents" \  # 메타데이터
--jsonmeta "path/to/metadata.json" \               # 메타데이터 json, --metadata보다 우선시됩니다.
--image "path/to/image/file" \                     # 이미지 파일
```

메타데이터 추출

```bash
fourier parse "path/to/fourierAnalogue.frad" \
\  # 선택 사항
--output "path/for/metadata" \  # Output file.meta.json, Output file.meta.image
```

ECC 패킹/리패킹

```bash
fourier ecc "path/to/fourierAnalogue.frad" \
\  # 선택 사항
--verbose
```

소프트웨어 녹음

```bash
fourier record "path/to/fourierAnalogue.frad" \
\  # 선택 사항
--bits 32 \                                        # 비트 심도
--frame_size 2048 \                                # 블록당 샘플 수
--enable_ecc \                                     # ECC 활성화 여부
--data_ecc_size 128 20 \                           # 데이터 블록과 ECC 블록의 크기
--little_endian \                                  # 엔디언
--metadata "Metadata Title" "Metadata contents" \  # 메타데이터
--jsonmeta "path/to/metadata.json" \               # 메타데이터 json, --metadata보다 우선시됩니다.
--image "path/to/image/file" \                     # 이미지 파일
```

메타데이터 JSON

```json
[
    {"key": "키",                  "type": "string", "value": "값"},
    {"key": "원작자",               "type": "string", "value": "한울"},
    {"key": "키와 String타입 인코딩", "type": "string", "value": "UTF-8"},
    {"key": "Base64 지원",         "type": "base64", "value": "QmFzZTY0IOyYiOyLnA=="},
    {"key": "파일 지원",            "type": "base64", "value": "7LWc64yAIDI1NlRpQuq5jOyngCDsp4Dsm5A="},
    {"key": "미지원 글자 없음",       "type": "string", "value": "유니코드에 있는 어떤 글자라도 호환됩니다!"},
    {"key": "중복 키 지원",          "type": "string", "value": "중복 키를 넣으면?"},
    {"key": "중복 키 지원",          "type": "string", "value": "짠!"}
]
```

## 기여 방법

레포지토리에서 새 브랜치를 만들고, 수정하고, 저에게 Merge 요청으로 심사를 받으시면 됩니다. 사실 FrAD 포맷 표준과 호환되기만 하면 웬만하면 다 통과됩니다.

## 개발자 정보

한울, <jun061119@proton.me>
