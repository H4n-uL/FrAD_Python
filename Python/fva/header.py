import struct
from .tools.header import header

b3_to_bits = {
    0b110: 512,
    0b101: 256,
    0b100: 128,
    0b011: 64,
    0b010: 32,
    0b001: 16
}

class header:
    def parse(file_path):
        d = dict()
        with open(file_path, 'rb') as f:
            header = f.read(256)

            d['signature'] = header[0x0:0xa]
            d['headlen'] = struct.unpack('<Q', header[0xa:0x12])[0]
            d['samprate'] = int.from_bytes(header[0x12:0x15], 'little')
            cfb = struct.unpack('<B', header[0x15:0x16])[0]
            d['channel'] = cfb >> 3
            d['bitrate'] = b3_to_bits.get(cfb & 0b111, None)
            d['isecc'] = struct.unpack('<B', header[0x16:0x17])[0] >> 7

            blocks = f.read(d['headlen'] - 256)
            i = 0
            while i < len(blocks):
                block_type = blocks[i:i+2]
                if block_type == b'\x72\xeb':
                    block_length = int.from_bytes(blocks[i+2:i+8], 'little')
                    title_length = int(struct.unpack('<I', blocks[i+8:i+12])[0])
                    title = blocks[i+12:i+12+title_length].decode('utf-8')
                    data = blocks[i+12+title_length:block_length].decode('utf-8')
                    d[title] = data
                    i += block_length
                else:
                    block_length = int(struct.unpack('<I', blocks[i+2:i+6])[0])
                    block_data = blocks[i+5:i+block_length].decode('utf-8')
                    d[block_type] = block_data
                    i += block_length

        return d

    def modify(file_path,
                title: str = None, lyrics: str = None, artist: str = None,
                album: str = None, track_number: int = None, genre: str = None,
                date: str = None, description: str = None, comment: str = None,
                composer: str = None, copyright: str = None, license: str = None,
                organization: str = None, location: str = None, performer: str = None,
                isrc: str = None, img: bytes = None):
        with open(file_path, 'rb') as f:
                head = f.read(256)

                header_length_old = struct.unpack('<Q', head[0xa:0x12])[0]
                sample_rate = head[0x12:0x15]
                cfb = struct.unpack('<B', head[0x15:0x16])[0]

                channel = cfb >> 3
                bits = b3_to_bits.get(cfb & 0b111)

                f.seek(header_length_old)
                audio = f.read()

                head_new = header.builder(sample_rate, channel, bits,
                title, lyrics, artist,
                album, track_number, genre,
                date, description, comment,
                composer, copyright, license,
                organization, location, performer,
                isrc, img)

                with open(file_path, 'wb') as f:
                    f.write(head_new)
                    f.write(audio)
