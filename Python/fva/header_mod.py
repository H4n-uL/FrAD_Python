import struct
from .header.header import header

b3_to_bits = {
    0b110: 512,
    0b101: 256,
    0b100: 128,
    0b011: 64,
    0b010: 32,
    0b001: 16
}

class headmod:
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
