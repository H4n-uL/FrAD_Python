from .comment_block import cb
import struct

bits_to_b3 = {
    512: 0b110,
    256: 0b101,
    128: 0b100,
    64: 0b011,
    32: 0b010,
    16: 0b001
}

class headb:
    def uilder(
            # Fixed Header
            sample_rate_bytes: bytes, channel: int,
            bits: int, isecc: bool, md5: bytes,

            # Metadata
            meta = None, img: bytes = None):
        b3 = bits_to_b3.get(bits, 0b000)

        cfb = ((channel-1) << 3) | b3

        signature = b'\x7e\x8b\xab\x89\xea\xc0\x9d\xa9\x68\x80'
        length = b'\x00'*8; sample_rate_bytes
        cfb_struct = struct.pack('<B', cfb)
        isecc = (0b1 if isecc else 0b0) << 7
        ecc_bits = struct.pack('<B', isecc | 0b0000000)
        reserved = b'\x00'*217

        blocks = bytes()

        if meta is not None:
            for i in range(len(meta)):
                blocks += cb.comment(meta[i][0], meta[i][1])
        if img is not None: blocks += cb.image(img)

        length = struct.pack('<Q', (len(signature + length + sample_rate_bytes + cfb_struct + ecc_bits + reserved + md5 + blocks)))

        header = signature + length + sample_rate_bytes + cfb_struct + ecc_bits + reserved + md5 + blocks
        return header
