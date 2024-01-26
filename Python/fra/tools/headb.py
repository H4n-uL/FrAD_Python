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
    def encode_efb(isecc, bits):
        ecc = (0b1 if isecc else 0b0) << 4
        b3 = bits_to_b3.get(bits, 0b000)
        return struct.pack('<B', ecc | b3)

    def decode_efb(efb):
        ecc = True if (efb >> 4 & 0b1) == 0b1 else False    # 0x08@0b100:    ECC Toggle(Enabled if 1)
        float_bits = efb & 0b111                            # 0x08@0b010-3b: Stream bit depth
        return ecc, float_bits

    def uilder(
            # Fixed Header
            sample_rate: bytes, channel: int,
            md5: bytes,

            # Metadata
            meta = None, img: bytes = None):

        signature = b'\x16\xb0\x03'

        channel_block = struct.pack('<B', channel - 1)
        sample_block = struct.pack('>I', sample_rate)

        blocks = bytes()

        if meta is not None:
            for i in range(len(meta)):
                blocks += cb.comment(meta[i][0], meta[i][1])
        if img is not None: blocks += cb.image(img)

        length = struct.pack('>Q', (256 + len(blocks)))

        header = signature + channel_block + sample_block + length + (b'\x00'*224) + md5 + blocks
        return header
