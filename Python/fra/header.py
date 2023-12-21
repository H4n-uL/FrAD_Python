from .common import methods
import struct
from .tools.headb import headb

b3_to_bits = {
    0b100: 128,
    0b011: 64,
    0b010: 32,
    0b001: 16
}

class header:
    def parse(file_path):
        d = list()
        with open(file_path, 'rb') as f:
            header = f.read(256)

            methods.signature(header[0x0:0x3])
            headlen = struct.unpack('>Q', header[0x8:0x10])[0]
            blocks = f.read(headlen - 256)
            i = j = 0
            image = b''
            while i < len(blocks):
                block_type = blocks[i:i+2]
                if block_type == b'\xfa\xaa':
                    block_length = int.from_bytes(blocks[i+2:i+8], 'big')
                    title_length = int(struct.unpack('<I', blocks[i+8:i+12])[0])
                    title = blocks[i+12:i+12+title_length].decode('utf-8')
                    data = blocks[i+12+title_length:i+block_length]
                    d.append([title, data])
                    i += block_length; j += 1
                elif block_type == b'\xf5\x55':
                    block_length = int(struct.unpack('>Q', blocks[i+2:i+10])[0])
                    data = blocks[i+10:i+block_length]
                    image = data
                    i += block_length
        return d, image

    def modify(file_path, meta = None, img: bytes = None):
        with open(file_path, 'r+b') as f:
            head = f.read(256)

            channel = struct.unpack('<B', head[0x3:0x4])[0] + 1
            sample_rate = struct.unpack('>I', head[0x4:0x8])[0]

            header_length = struct.unpack('>Q', head[0x8:0x10])[0]
            efb = struct.unpack('<B', head[0x10:0x11])[0]
            is_ecc_on = True if (efb >> 4 & 0b1) == 0b1 else False
            checksum_header = head[0xf0:0x100]

            bits = b3_to_bits.get(efb & 0b111)

            f.seek(header_length)
            audio = f.read()

            head_new = headb.uilder(sample_rate, channel, bits, is_ecc_on, checksum_header,
            meta, img)

            file = head_new + audio
            f.write(file)
