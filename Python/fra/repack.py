from .common import methods
import hashlib
from .tools.ecc import ecc
import struct

class repack:
    def ecc(file_path):
        with open(file_path, 'rb') as f:
            head = f.read(256)

            methods.signature(head[0x0:0x3])

            header_length = struct.unpack('>Q', head[0x8:0x10])[0]
            efb = struct.unpack('<B', head[0x10:0x11])[0]
            is_ecc_on = True if (efb >> 4) == 0b1 else False

            f.seek(header_length)

            block = f.read()

            if is_ecc_on:
                block = ecc.decode(block)

            data = ecc.encode(block, True)
            checksum = hashlib.md5(data).digest()

            f.seek(0)
            head = bytearray(f.read(header_length))
            head[0xf0:0x100] = checksum
            head[0x10:0x11] = struct.pack('<B', 0b1 << 4 | efb)
            file = bytes(head) + data

        with open(file_path, 'wb') as f:
            f.write(file)
