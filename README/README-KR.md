# Fourier Analogue-in-Digital

## 프로젝트 개요

[AAPM](https://mikhael-openworkspace.notion.site/Project-Archivist-e512fa7a21474ef6bdbd615a424293cf)@Audio-8151의 Python 구현체입니다. 자세한 내용은 [Notion](https://mikhael-openworkspace.notion.site/Fourier-Analogue-in-Digital-d170c1760cbf4bb4aaea9b1f09b7fead?pvs=4)에서 확인하실 수 있습니다.

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

## 메타데이터 JSON 예시

```json
[
    {"key": "키",                  "type": "string", "value": "값"},
    {"key": "원작자",               "type": "string", "value": "한울"},
    {"key": "키와 String타입 인코딩", "type": "string", "value": "UTF-8"},
    {"key": "Base64 지원",         "type": "base64", "value": "QmFzZTY0IOyYiOyLnA=="},
    {"key": "파일 지원",            "type": "base64", "value": "7LWc64yAIDI1NlRpQuq5jOyngCDsp4Dsm5A="},
    {"key": "미지원 글자 없음",       "type": "string", "value": "유니코드에 있는 어떤 글자라도 호환됩니다!"},
    {"key": "중복 키 지원",          "type": "string", "value": "중복 키를 넣으면?"},
    {"key": "중복 키 지원",          "type": "string", "value": "짠!"},
    {"key": "",                   "type": "string", "value": "키 없는 메타데이터도 지원"}
]
```

## 기여 방법

레포지토리에서 새 브랜치를 만들고, 수정하고, 저에게 Merge 요청으로 심사를 받으시면 됩니다. 사실 FrAD 포맷 표준과 호환되기만 하면 웬만하면 다 통과됩니다.

## 구현 요구 사항

1. 필수 구현

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

2. 선택 구현

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
    ...과 온갖 잡다한 기능들
    ```

## 개발자 정보

한울, <jun061119@proton.me>
