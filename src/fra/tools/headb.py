from .comment_block import cb
import struct

bits_to_b3 = {
    128: 0b110,
    64: 0b101,
    48: 0b100,
    32: 0b011,
    24: 0b010,
    16: 0b001
}

class headb:
    def encode_efb(lossy, isecc, big_endian, bits):
        lossy = (lossy and 0b1 or 0b0) << 5
        ecc = (isecc and 0b1 or 0b0) << 4
        endian = (big_endian and 0b1 or 0b0) << 3
        b3 = bits_to_b3.get(bits, 0b000)
        return struct.pack('<B', lossy | ecc | endian | b3)

    def decode_efb(efb):
        lossy = efb>>5&0b1==0b1 and True or False      # 0x08@0b100:    ECC Toggle(Enabled if 1)
        ecc = efb>>4&0b1==0b1 and True or False        # 0x08@0b100:    ECC Toggle(Enabled if 1)
        big_endian = efb>>3&0b1==0b1 and True or False # 0x08@0b011:    Endian
        float_bits = efb & 0b111                       # 0x08@0b010-3b: Stream bit depth
        return lossy, ecc, big_endian, float_bits

    def uilder(meta = None, img: bytes = None):

        signature = b'fRad'

        blocks = bytes()

        if meta is not None and meta != []:
            for i in range(len(meta)):
                blocks += cb.comment(meta[i][0], meta[i][1])
        if img is not None and img != b'': blocks += cb.image(img)

        length = struct.pack('>Q', (64 + len(blocks)))

        header = signature + (b'\x00'*4) + length + (b'\x00'*48) + blocks
        return header
