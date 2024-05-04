from .comment_block import cb
import struct

class headb:
    @staticmethod
    def encode_efb(profile, isecc, little_endian, bits):
        profile = profile << 5
        ecc = (isecc and 0b1 or 0b0) << 4
        endian = (little_endian and 0b1 or 0b0) << 3
        return struct.pack('<B', profile | ecc | endian | bits)

    @staticmethod
    def decode_efb(efb):
        profile = efb>>5                                  # 0x08@0b111-3b: ECC Toggle(Enabled if 1)
        ecc = efb>>4&0b1==0b1 and True or False           # 0x08@0b100:    ECC Toggle(Enabled if 1)
        little_endian = efb>>3&0b1==0b1 and True or False # 0x08@0b011:    Endian
        float_bits = efb & 0b111                          # 0x08@0b010-3b: Stream bit depth
        return profile, ecc, little_endian, float_bits

    @staticmethod
    def uilder(meta: list[list[str]] | None = None, img: bytes | None = None):

        signature = b'fRad'

        blocks = bytes()

        if meta is not None and meta != []:
            for i in range(len(meta)):
                blocks += cb.comment(meta[i][0], meta[i][1])
        if img is not None and img != b'': blocks += cb.image(img)

        length = struct.pack('>Q', (64 + len(blocks)))

        header = signature + (b'\x00'*4) + length + (b'\x00'*48) + blocks
        return header
