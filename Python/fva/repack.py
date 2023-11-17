from .tools.ecc import ecc
from .header import header
import struct

class repack:
    def ecc(file_path, apply_ecc: bool = False):
        with open(file_path, 'r+b') as f:
            head = f.read(256)

            signature = head[0x0:0xa]
            if signature != b'\x7e\x8b\xab\x89\xea\xc0\x9d\xa9\x68\x80':
                raise Exception('This is not Fourier Analogue file.')

            header_length = struct.unpack('<Q', head[0xa:0x12])[0]
            is_ecc_on = True if (struct.unpack('<B', header[0x16:0x17])[0] >> 7) == 0b1 else False

            f.seek(header_length)

            block = f.read()
            block = ecc.decode(block, is_ecc_on)

            f.seek(0)
            head = bytearray(f.read(header_length))
            apply_ecc = 0b1 if apply_ecc else 0b0 << 7
            head[0x16:0x17] = struct.pack('<B', apply_ecc | 0b0000000)
            head = bytes(head)
            data = ecc.encode(ecc.decode(block), apply_ecc)
            f.write(head)
            f.write(data)
