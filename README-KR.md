# Fourier Analogue-in-Digital

## 프로젝트 개요

Project Archivist의 일환으로 아날로그 신호를 그대로 디지털 속에 담아낸다는 목표를 가지고 개발되었습니다. 자세한 내용은 [Notion](https://mikhael-openworkspace.notion.site/Project-Archivist-e512fa7a21474ef6bdbd615a424293cf)에서 확인하실 수 있습니다.

주의: 아카이브의 목적은 언제나 여러분이 쉽게 중요한 데이터에 접근하게 하기 위함입니다. 중요한 음원 파일은 꼭 백업해 두시기 바랍니다.

## 설치 방법

1. Git zip 다운로드
2. Python(3.11^) 설치
3. install.sh 실행
4. source ~/.*shrc로 shell 재실행

설치는 아직 유닉스 계열 OS에서만 지원합니다.

## 사용 방법

인코딩

```bash
fourier encode "path/to/audio.flac" \
--bits 32 \                                        # 비트 심도
\  # 선택 사항
--output "path/to/fourierAnalogue.frad" \          # 출력 파일
--samples_per_block 2048 \                         # 블록당 샘플 수
--enable_ecc \                                     # ECC 활성화 여부
--data_ecc_size 128 20 \                           # 데이터 블록과 ECC 블록의 크기
--big_endian \                                     # 엔디언
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

.json 파일의 예시는 Examples 폴더에 들어 있습니다.

## 기여 방법

![몰?루](https://item.kakaocdn.net/do/4a675e36e71c3538c5e7ada87a2b28fef43ad912ad8dd55b04db6a64cddaf76d)

## 라이선스

### 적용 대상

본 라이선스는 Fourier Analogue-in-Digital의 저작인격권, 지식재산권, Git 레포지토리, 소스 코드 등 전체에 적용됩니다.

### 허가되는 것들

모든 사람은 소스 코드의 사용과 재가공, 재배포, 개선 및 기여, 프로젝트 구현 방식의 변경(단, 이 레포지토리의 소스 코드와 호환되어야 합니다.), 상업적 프로그램의 일부로 활용을 할 수 있습니다.

### 제한되는 것들

이 레포지토리를 단독으로 상업적으로 사용하는 것은 금지됩니다.

프로젝트의 원저작자는 명시하지 않아도 됩니다. 단, 기여를 하지 않은 제3자가 자신이 이 프로젝트의 저작권을 주장하는 것은 금지됩니다. 기여를 한 경우, 원저작자와 기여자의 이름을 병기할 수 있습니다.

이 프로젝트의 핵심 개념으로 특허권이나 상표권을 주장할 수 없습니다. 단, 부가기능을 개발하여 적용한 경우 해당 부가기능에 한하여 독립적인 특허권 또는 상표권을 주장할 수 있습니다.

DRM의 적용은 엄격히 금지됩니다. 모두에게 열려 있는 포맷인 Fourier Analogue-in-Digital에 DRM을 적용하는 것은 용납할 수 없습니다.

### 면책 사항

이 소스 코드를 사용함으로써 발생하는 어떠한 결과에 대해서도 원저작자와 기여자들은 책임을 지지 않습니다. 사실 책임을 지고 싶어도 질 수 없습니다.

## 개발자 정보

한울, <jun061119@proton.me>
