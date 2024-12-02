import math, struct
from ..fourier.profiles import compact, COMPACT
from ..common import FRM_SIGN, crc16_ansi
from zlib import crc32

def encode_pfb(profile: int, isecc: bool, little_endian: bool, bits: int) -> bytes:
    profile <<= 5
    ecc = (isecc and 0b1 or 0b0) << 4
    endian = (little_endian and 0b1 or 0b0) << 3
    return struct.pack('<B', profile | ecc | endian | bits)

def decode_pfb(pfb: bytes) -> tuple[int, bool, bool, int]:
    pfbint = struct.unpack('<B', pfb)[0]
    profile = pfbint>>5                                  # 0x08@0b111-3b: ECC Toggle(Enabled if 1)
    ecc = pfbint>>4&0b1==0b1 and True or False           # 0x08@0b100:    ECC Toggle(Enabled if 1)
    little_endian = pfbint>>3&0b1==0b1 and True or False # 0x08@0b011:    Endian
    float_bits = pfbint & 0b111                          # 0x08@0b010-3b: Stream bit depth
    return profile, ecc, little_endian, float_bits

def encode_css_prf1(channels: int, srate: int, fsize: int, force_flush: bool) -> bytes:
    chnl = (channels-1)<<10
    srate = compact.get_srate_index(srate) << 6
    fsize = min((x for x in compact.SAMPLES_LI if x >= fsize), default=0)
    mult = next((key for key, values in compact.SAMPLES.items() if fsize in values), 0)
    px = tuple(compact.SAMPLES).index(mult) << 4
    fsize = int(math.log2(fsize / mult)) << 1

    return struct.pack('>H', chnl | srate | px | fsize | force_flush)

def decode_css_prf1(css: bytes) -> tuple[int, int, int, bool]:
    cssint = struct.unpack('>H', css)[0]
    channels = (cssint>>10) + 1                    # 0x09@0b111-6b: Channels
    srate = compact.SRATES[cssint>>6&0b1111]       # 0x09@0b001-4b: Sample rate index
    fsize_prefix = [128, 144, 192][cssint>>4&0b11] # 0x0a@0b101-2b: Frame size prefix
    fsize = fsize_prefix * 2**(cssint>>1&0b111)    # 0x0a@0b011-3b: Frame size
    return channels, srate, fsize, cssint & 0b1

class ASFH:
    def __init__(self):
        self.frmbytes = 0
        self.buffer = b''
        self.all_set = False
        self.header_bytes = 0

        self.endian, self.bit_depth_index = False, 0
        self.channels, self.srate, self.fsize = 0, 0, 0

        self.ecc, self.ecc_dsize, self.ecc_codesize = False, 0, 0
        self.profile, self.overlap_ratio = 0, 0
        self.crc = b''

    def criteq(self, other: 'ASFH') -> bool:
        return self.channels == other.channels and self.srate == other.srate

    def write(self, frad: bytes) -> bytes:
        fhead = FRM_SIGN

        fhead += struct.pack('>I', len(frad))
        fhead += encode_pfb(self.profile, self.ecc, self.endian, self.bit_depth_index)

        if self.profile in COMPACT:
            fhead += encode_css_prf1(self.channels, self.srate, self.fsize, False)
            fhead += struct.pack('B', max(self.overlap_ratio - 1, 0))
            if self.ecc:
                fhead += struct.pack('BB', self.ecc_dsize, self.ecc_codesize)
                fhead += crc16_ansi(frad).to_bytes(2, 'big')
        else:
            fhead += struct.pack('B', self.channels-1)
            fhead += struct.pack('BB', self.ecc_dsize, self.ecc_codesize)
            fhead += struct.pack('>I', self.srate)
            fhead += b'\x00'*8
            fhead += struct.pack('>I', self.fsize)
            fhead += crc32(frad).to_bytes(4, 'big')

        frad = fhead + frad

        return frad

    def force_flush(self) -> bytes:
        fhead = FRM_SIGN
        fhead += b'\x00'*4
        fhead += encode_pfb(self.profile, self.ecc, self.endian, self.bit_depth_index)

        if self.profile in COMPACT:
            fhead += encode_css_prf1(1, 96000, 128, True)
            fhead += b'\x00'
        else: return b''

        return fhead

    def fill_buffer(self, buffer: bytes, target_size: int) -> tuple[bool, bytes]:
        if len(self.buffer) < target_size:
            cutout = target_size - len(self.buffer)
            self.buffer += buffer[:cutout]
            buffer = buffer[cutout:]
            if len(self.buffer) < target_size: return False, buffer
        self.header_bytes = target_size
        return True, buffer

    def read(self, buffer: bytes) -> tuple[str, bytes]:
        x, buffer = self.fill_buffer(buffer, 9)
        if not x: return 'Incomplete', buffer
        self.frmbytes = struct.unpack('>I', self.buffer[4:8])[0]
        self.profile, self.ecc, self.endian, self.bit_depth_index = decode_pfb(self.buffer[8:9])

        if self.profile in COMPACT:
            x, buffer = self.fill_buffer(buffer, 12)
            if not x: return 'Incomplete', buffer
            self.channels, self.srate, self.fsize, force_flush = decode_css_prf1(self.buffer[9:11])
            if force_flush: return 'ForceFlush', buffer

            self.overlap_ratio = self.buffer[11]
            if self.overlap_ratio != 0: self.overlap_ratio += 1

            if self.ecc:
                x, buffer = self.fill_buffer(buffer, 16)
                if not x: return 'Incomplete', buffer
                self.ecc_dsize, self.ecc_codesize = struct.unpack('BB', self.buffer[12:14])
                self.crc = self.buffer[14:16]

        else:
            x, buffer = self.fill_buffer(buffer, 32)
            if not x: return 'Incomplete', buffer
            self.channels = self.buffer[9] + 1
            self.ecc_dsize, self.ecc_codesize = struct.unpack('BB', self.buffer[10:12])
            self.srate = struct.unpack('>I', self.buffer[12:16])[0]
            self.fsize = struct.unpack('>I', self.buffer[24:28])[0]
            self.crc = self.buffer[28:32]

        if self.frmbytes == 0xFFFFFFFF:
            x, buffer = self.fill_buffer(buffer, self.header_bytes + 8)
            if not x: return 'Incomplete', buffer
            self.frmbytes = struct.unpack('>Q', self.buffer[-8:])[0]

        self.all_set = True
        return 'Complete', buffer

    def clear(self):
        self.all_set = False
        self.buffer = b''
