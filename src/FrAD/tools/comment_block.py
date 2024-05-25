import struct

IMAGE =   b'\xf5'
COMMENT = b'\xfa\xaa'

class cb:
    @staticmethod
    def comment(title: str, data: str | bytes) -> bytes:
        if type(data) == bytes: dbytes = data
        elif type(data) == str: dbytes = data.encode('utf-8')
        title_length = struct.pack('>I', len(title))
        data_comb = title.encode('utf-8') + dbytes
        block_length = (len(data_comb) + 12).to_bytes(6, 'big')
        return bytes(COMMENT + block_length + title_length + data_comb)

    @staticmethod
    def image(data: bytes, pictype: int = 3) -> bytes:
        if pictype not in range(0, 21): pictype = 3
        apictype = struct.pack('<B', 0b01000000 | pictype)
        block_length = struct.pack('>Q', len(data) + 10)
        return bytes(IMAGE + apictype + block_length + data)
