from ..common import methods
import base64, struct

IMAGE =   b'\xf5'
COMMENT = b'\xfa\xaa'

class metablock:
    @staticmethod
    def comment(title: str, data: str | bytes) -> bytes:
        if type(data) == bytes: dbytes = data
        elif type(data) == str: dbytes = data.encode('utf-8')
        title_length = struct.pack('>I', len(title))
        data_comb = title.encode('utf-8') + dbytes
        block_length = (len(data_comb) + 12).to_bytes(6, 'big')
        return bytes(COMMENT + block_length + title_length + data_comb)

    @staticmethod
    def image(data: bytes, pictype: int = 3) -> bytes:
        if pictype not in range(0, 21): pictype = 3
        apictype = struct.pack('<B', 0b01000000 | pictype)
        block_length = struct.pack('>Q', len(data) + 10)
        return bytes(IMAGE + apictype + block_length + data)

class headb:
    @staticmethod
    def encode_pfb(profile: int, isecc: bool, little_endian: bool, bits: int) -> bytes:
        profile = profile << 5
        ecc = (isecc and 0b1 or 0b0) << 4
        endian = (little_endian and 0b1 or 0b0) << 3
        return struct.pack('<B', profile | ecc | endian | bits)

    @staticmethod
    def decode_pfb(pfb: int) -> tuple[int, bool, bool, int]:
        profile = pfb>>5                                  # 0x08@0b111-3b: ECC Toggle(Enabled if 1)
        ecc = pfb>>4&0b1==0b1 and True or False           # 0x08@0b100:    ECC Toggle(Enabled if 1)
        little_endian = pfb>>3&0b1==0b1 and True or False # 0x08@0b011:    Endian
        float_bits = pfb & 0b111                          # 0x08@0b010-3b: Stream bit depth
        return profile, ecc, little_endian, float_bits

    @staticmethod
    def uilder(meta: list[list[str]] | None = None, img: bytes | None = None):
        signature = b'fRad'
        blocks = bytes()

        if meta:
            for i in range(len(meta)): blocks += metablock.comment(meta[i][0], meta[i][1])
        if img: blocks += metablock.image(img)

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
