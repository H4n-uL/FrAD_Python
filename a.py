from unireedsolomon import rs

codec = rs.RSCoder(255, 245)  # 10개의 에러 정정 코드를 생성합니다.
encoded = codec.encode(b'Hello, world!')  # 'Hello, world!'를 인코딩합니다.
print(encoded)

try:
    decoded = codec.decode(encoded)  # 인코딩된 데이터를 디코딩합니다.
    print(decoded)
except rs.RSCodecError as e:
    print(f'Decoding failed: {e}')