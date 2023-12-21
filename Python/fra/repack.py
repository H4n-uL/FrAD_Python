from .common import methods
import hashlib
from .tools.ecc import ecc
import struct

class repack:
    def ecc(file_path):
        with open(file_path, 'r+b') as f:
            head = f.read(256)

            methods.signature(head[0x0:0x3])

            header_length = struct.unpack('>Q', head[0x8:0x10])[0]
            is_ecc_on = True if (struct.unpack('<B', head[0x16:0x17])[0] >> 7) == 0b1 else False

            f.seek(header_length)

            block = f.read()

            if is_ecc_on:
                block = ecc.decode(block)

            data = ecc.encode(block, True)
            checksum = hashlib.md5(data).digest()

            f.seek(0)
            head = bytearray(f.read(header_length))
            head[0xf0:0x100] = checksum
            head[0x16:0x17] = struct.pack('<B', 0b1 << 7)
            file = bytes(head) + data

            f.seek(0)
            f.write(file)
