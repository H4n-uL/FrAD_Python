import hashlib
from .tools.ecc import ecc
from .header import header
import struct

class repack:
    def ecc(file_path):
        with open(file_path, 'r+b') as f:
            head = f.read(256)

            signature = head[0x0:0xa]
            if signature != b'\x7e\x8b\xab\x89\xea\xc0\x9d\xa9\x68\x80':
                raise Exception('This is not Fourier Analogue file.')

            header_length = struct.unpack('<Q', head[0xa:0x12])[0]

            f.seek(header_length)

            block = f.read()

            data = ecc.encode(ecc.decode(block), True)
            checksum = hashlib.md5(data).digest()

            f.seek(0)
            head = bytearray(f.read(header_length))
            head[0xf0:0x100] = checksum
            head = bytes(head)

            f.seek(0)
            f.write(head)
            f.write(data)
