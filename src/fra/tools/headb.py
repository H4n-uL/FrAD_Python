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
    def encode_efb(isecc, big_endian, bits):
        ecc = (0b1 if isecc else 0b0) << 4
        endian = (0b1 if big_endian else 0b0) << 3
        b3 = bits_to_b3.get(bits, 0b000)
        return struct.pack('<B', ecc | endian | b3)

    def decode_efb(efb):
        ecc = True if (efb >> 4 & 0b1) == 0b1 else False        # 0x08@0b100:    ECC Toggle(Enabled if 1)
        big_endian = True if (efb >> 3 & 0b1) == 0b1 else False # 0x08@0b011:    Endian
        float_bits = efb & 0b111                                # 0x08@0b010-3b: Stream bit depth
        return ecc, big_endian, float_bits

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
