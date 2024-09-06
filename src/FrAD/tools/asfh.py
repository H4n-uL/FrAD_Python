from ..profiles.prf import profiles
import struct, io
from .headb import headb
from ..common import variables, methods
import zlib

class ASFH:
    def __init__(self):
            self.total_bytes, self.frmbytes = 0, 0

            self.endian, self.float_bits = False, 0
            self.chnl, self.srate, self.fsize = 0, 0, 0

            self.ecc, self.ecc_dsize, self.ecc_codesize = False, 0, 0
            self.profile, self.overlap = 0, 0
            self.crc = b''

    def update(self, file: io.BufferedReader) -> bool:
        fhead = variables.FRM_SIGN + file.read(5)
        self.frmbytes = struct.unpack('>I', fhead[0x4:0x8])[0]        # 0x04-4B: Audio Stream Frame length
        self.profile, self.ecc, self.endian, self.float_bits = headb.decode_pfb(fhead[0x8:0x9]) # 0x08: EFloat Byte

        if self.profile in profiles.LOSSLESS:
            fhead += file.read(23)
            self.chnl = struct.unpack('>B', fhead[0x9:0xa])[0] + 1     # 0x09:    Channels
            self.ecc_dsize = struct.unpack('>B', fhead[0xa:0xb])[0]    # 0x0a:    ECC Data block size
            self.ecc_codesize = struct.unpack('>B', fhead[0xb:0xc])[0] # 0x0b:    ECC Code size
            self.srate = struct.unpack('>I', fhead[0xc:0x10])[0]       # 0x0c-4B: Sample rate
            self.fsize = struct.unpack('>I', fhead[0x18:0x1c])[0]      # 0x18-4B: Samples in a frame per channel
            self.crc = fhead[0x1c:0x20]                                # 0x1c-4B: ISO 3309 CRC32 of Audio Data

        if self.profile in profiles.COMPACT:
            fhead += file.read(3)
            self.chnl, self.srate, self.fsize, force_flush = headb.decode_css_prf1(fhead[0x9:0xb])
            if force_flush: return True
            self.overlap = struct.unpack('>B', fhead[0xb:0xc])[0]      # 0x0b: Overlap rate
            if self.overlap != 0: self.overlap += 1
            if self.ecc == True:
                fhead += file.read(4)
                self.ecc_dsize = struct.unpack('>B', fhead[0xc:0xd])[0]
                self.ecc_codesize = struct.unpack('>B', fhead[0xd:0xe])[0]
                self.crc = fhead[0xe:0x10]                             # 0x0e-2B: ANSI CRC16 of Audio Data

        if self.frmbytes == variables.FRM_MAXSZ:
            fhead += file.read(8)
            self.frmbytes = struct.unpack('>Q', fhead[-8:])[0]

        self.headlen = len(fhead)
        return False

    def write_frame(self, file: io.BufferedWriter, frame: bytes) -> int:
        if not self.ecc: self.ecc_dsize, self.ecc_codesize = 0, 0
        data = bytes(
            variables.FRM_SIGN +
            struct.pack('>I', min(len(frame), variables.FRM_MAXSZ)) +
            headb.encode_pfb(self.profile, self.ecc, self.endian, self.float_bits)
        )
        if self.profile in profiles.LOSSLESS:
            data += (
                struct.pack('>B', self.chnl - 1) +
                struct.pack('>B', self.ecc_dsize) +
                struct.pack('>B', self.ecc_codesize) +
                struct.pack('>I', self.srate) +
                b'\x00'*8 +
                struct.pack('>I', self.fsize) +
                struct.pack('>I', zlib.crc32(frame))
            )
        elif self.profile in profiles.COMPACT:
            data += (
                headb.encode_css_prf1(self.chnl, self.srate, self.fsize, False) +
                struct.pack('>B', min(max(self.overlap, 1) - 1, 255))
            )
            if self.ecc:
                data += (
                    struct.pack('>B', self.ecc_dsize) +
                    struct.pack('>B', self.ecc_codesize) +
                    struct.pack('>H', methods.crc16_ansi(frame))
                )
        if len(frame) >= variables.FRM_MAXSZ: data += struct.pack('>Q', len(frame))
        data += frame
        file.write(data)
        return len(data)
    
    def flush(self, file: io.BufferedWriter) -> None:
        data = bytes(
            variables.FRM_SIGN +
            struct.pack('>I', 0) +
            headb.encode_pfb(self.profile, self.ecc, self.endian, self.float_bits)
        )

        if self.profile in profiles.COMPACT:
            data += (
                headb.encode_css_prf1(self.chnl, self.srate, self.fsize, True) +
                struct.pack('>B', 0)
            )
        else: return

        file.write(data)