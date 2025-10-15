SIGNATURE = b'fRad'
FRM_SIGN = b'\xff\xd0\xd2\x98'

crc16t_ansi = [(lambda c: [c := (c >> 1) ^ 0xA001 if c & 0x0001 else c >> 1 for _ in range(8)][-1])(i) for i in range(256)]

def crc16_ansi(data: bytes):
    crc = 0
    for byte in data:
        crc = (crc >> 8) ^ crc16t_ansi[(crc ^ byte) & 0xff]
    return crc
