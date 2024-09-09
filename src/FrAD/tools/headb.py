from ..common import variables, methods
from ..profiles.prf import compact
import base64, math, struct

IMAGE =   b'\xf5'
COMMENT = b'\xfa\xaa'

class metablock:
    @staticmethod
    def comment(title: str, data: str | bytes) -> bytes:
        if isinstance(data, bytes): dbytes = data
        elif isinstance(data, str): dbytes = data.encode('utf-8')
        else: dbytes = str(data).encode('utf-8')
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
        profile <<= 5
        ecc = (isecc and 0b1 or 0b0) << 4
        endian = (little_endian and 0b1 or 0b0) << 3
        return struct.pack('<B', profile | ecc | endian | bits)

    @staticmethod
    def decode_pfb(pfb: bytes) -> tuple[int, bool, bool, int]:
        pfbint = struct.unpack('<B', pfb)[0]
        profile = pfbint>>5                                  # 0x08@0b111-3b: ECC Toggle(Enabled if 1)
        ecc = pfbint>>4&0b1==0b1 and True or False           # 0x08@0b100:    ECC Toggle(Enabled if 1)
        little_endian = pfbint>>3&0b1==0b1 and True or False # 0x08@0b011:    Endian
        float_bits = pfbint & 0b111                          # 0x08@0b010-3b: Stream bit depth
        return profile, ecc, little_endian, float_bits

    @staticmethod
    def encode_css_prf1(channels: int, srate: int, fsize: int, force_flush: bool) -> bytes:
        chnl = (channels-1)<<10
        srate = compact.srates.index(srate) << 6
        fsize = min((x for x in compact.samples_li if x >= fsize), default=0)
        mult = next((key for key, values in compact.samples.items() if fsize in values), 0)
        px = tuple(compact.samples).index(mult) << 4
        fsize = int(math.log2(fsize / mult)) << 1

        return struct.pack('>H', chnl | srate | px | fsize | (force_flush and 0b1 or 0b0))

    @staticmethod
    def decode_css_prf1(css: bytes) -> tuple[int, int, int, bool]:
        cssint = struct.unpack('>H', css)[0]
        channels = (cssint>>10) + 1                    # 0x09@0b111-6b: Channels
        srate = compact.srates[cssint>>6&0b1111]       # 0x09@0b001-4b: Sample rate index
        fsize_prefix = [128, 144, 192][cssint>>4&0b11] # 0x0a@0b101-2b: Frame size prefix
        fsize = fsize_prefix * 2**(cssint>>1&0b111)    # 0x0a@0b011-3b: Frame size
        return channels, srate, fsize, cssint & 0b1

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
