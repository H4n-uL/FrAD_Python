from .tools.ecc import ecc
import struct

class repack:
    def ecc(file_path, ecc_type):
        with open(file_path, 'r+b') as f:
            header = f.read(256)

            signature = header[0x0:0xa]
            if signature != b'\x7e\x8b\xab\x89\xea\xc0\x9d\xa9\x68\x80':
                raise Exception('This is not Fourier Analogue file.')

            header_length = struct.unpack('<Q', header[0xa:0x12])[0]
            ecc_opt = struct.unpack('<B', header[0x16:0x17])[0] >> 5

            f.seek(header_length)

            block = f.read()

            f.seek(0)
            head = f.read(header_length)
            data = ecc.encode(ecc.decode(block, ecc_opt), ecc_type)
            f.write(head)
            f.write(data)
