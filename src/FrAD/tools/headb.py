from .comment_block import cb
from ..common import methods
import base64, struct

class headb:
    @staticmethod
    def encode_efb(profile: int, isecc: bool, little_endian: bool, bits: int) -> bytes:
        profile = profile << 5
        ecc = (isecc and 0b1 or 0b0) << 4
        endian = (little_endian and 0b1 or 0b0) << 3
        return struct.pack('<B', profile | ecc | endian | bits)

    @staticmethod
    def decode_efb(efb: bytes) -> tuple[int, bool, bool, int]:
        efbint = int.from_bytes(efb, 'big')
        profile = efbint>>5                                  # 0x08@0b111-3b: ECC Toggle(Enabled if 1)
        ecc = efbint>>4&0b1==0b1 and True or False           # 0x08@0b100:    ECC Toggle(Enabled if 1)
        little_endian = efbint>>3&0b1==0b1 and True or False # 0x08@0b011:    Endian
        float_bits = efbint & 0b111                          # 0x08@0b010-3b: Stream bit depth
        return profile, ecc, little_endian, float_bits

    @staticmethod
    def uilder(meta: list[list[str]] | None = None, img: bytes | None = None):
        signature = b'fRad'
        blocks = bytes()

        if meta:
            for i in range(len(meta)): blocks += cb.comment(meta[i][0], meta[i][1])
        if img: blocks += cb.image(img)

        length = struct.pack('>Q', (64 + len(blocks)))

        header = signature + (b'\x00'*4) + length + (b'\x00'*48) + blocks
        return header
    
    @staticmethod
    def parser(file_path: str) -> tuple[list[str], bytes | None]:
        meta, img = [], None
        with open(file_path, 'rb') as f:
            head = f.read(64)
            ftype = methods.signature(head[0x0:0x4])
            if ftype == 'container':
                while True:
                    block_type = f.read(2)
                    if not block_type: break
                    if block_type == b'\xfa\xaa':
                        block_length = int.from_bytes(f.read(6), 'big')
                        title_length = int(struct.unpack('>I', f.read(4))[0])
                        title = f.read(title_length).decode('utf-8')
                        data = f.read(block_length-title_length-12)
                        try: d = [title, data.decode('utf-8'), 'string']
                        except UnicodeDecodeError: d = [title, base64.b64encode(data).decode('utf-8'), 'base64']
                        meta.append(d)
                    elif block_type[0] == 0xf5:
                        block_length = int(struct.unpack('>Q', f.read(8))[0])
                        img = f.read(block_length-10)
                    elif block_type == b'\xff\xd0': break
            elif ftype == 'stream': return [], None
        return meta, img
