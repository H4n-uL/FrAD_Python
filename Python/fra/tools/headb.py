from .comment_block import cb
import struct

bits_to_b3 = {
    128: 0b100,
    64: 0b011,
    32: 0b010,
    16: 0b001
}

class headb:
    def uilder(
            # Fixed Header
            sample_rate: bytes, channel: int, 
            bits: int, isecc: bool, md5: bytes,

            # Metadata
            meta = None, img: bytes = None):
        b3 = bits_to_b3.get(bits, 0b000)

        signature = b'\x16\xb0\x03'

        channel_block = struct.pack('<B', channel - 1)
        sample_block = struct.pack('>I', sample_rate)

        length = b'\x00'*8
        ecc = (0b1 if isecc else 0b0) << 4
        efb_struct = struct.pack('<B', ecc | b3)
        isecc = (0b1 if isecc else 0b0) << 7
        ecc_bits = struct.pack('<B', isecc | 0b0000000)

        blocks = bytes()

        if meta is not None:
            for i in range(len(meta)):
                blocks += cb.comment(meta[i][0], meta[i][1])
        if img is not None: blocks += cb.image(img)

        length = struct.pack('>Q', (256 + len(blocks)))

        header = signature + channel_block + sample_block + length + efb_struct + (b'\x00'*5) + ecc_bits + (b'\x00'*217) + md5 + blocks
        return header
