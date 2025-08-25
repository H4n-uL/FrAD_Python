from . import common
from .fourier import profiles
from .tools import ecc
from .tools.asfh import ASFH
import zlib
import sys

class Repairer:
    def __init__(self, ecc_ratio: tuple[int, int] = (96, 24)):
        if ecc_ratio[0] == 0:
            print("ECC data size must not be zero", file=sys.stderr)
            print("Setting ECC to default 96 24", file=sys.stderr)
            ecc_ratio = (96, 24)
        if ecc_ratio[0] + ecc_ratio[1] > 255:
            print(f"ECC data size and check size must not exceed 255, given: {ecc_ratio[0]} and {ecc_ratio[1]}", file=sys.stderr)
            print("Setting ECC to default 96 24", file=sys.stderr)
            ecc_ratio = (96, 24)

        self.asfh = ASFH()
        self.buffer = b''

        self.fix_error = True
        self.ecc_ratio = ecc_ratio
        self.broken_frame = False

    def is_empty(self) -> bool: return len(self.buffer) < len(common.FRM_SIGN) or self.broken_frame

    def process(self, stream: bytes) -> bytes:
        self.buffer += stream
        ret = b''

        while True:
            if self.asfh.all_set:
                self.broken_frame = False
                if len(self.buffer) < self.asfh.frmbytes:
                    if len(stream) == 0: self.broken_frame = True
                    break

                frad, self.buffer = self.buffer[:self.asfh.frmbytes], self.buffer[self.asfh.frmbytes:]

                if self.asfh.ecc:
                    repair = self.fix_error and (
                        (self.asfh.profile in profiles.LOSSLESS and zlib.crc32(frad) != self.asfh.crc) or
                        (self.asfh.profile in profiles.COMPACT and common.crc16_ansi(frad) != self.asfh.crc)
                    )
                    frad = ecc.decode(frad, self.asfh.ecc_dsize, self.asfh.ecc_codesize, repair)

                frad = ecc.encode(frad, self.ecc_ratio[0], self.ecc_ratio[1])
                self.asfh.ecc = True
                self.asfh.ecc_dsize, self.asfh.ecc_codesize = self.ecc_ratio

                ret += self.asfh.write(frad)
                self.asfh.clear()
            else:
                if not self.asfh.buffer[:len(common.FRM_SIGN)] == common.FRM_SIGN:
                    i = self.buffer.find(common.FRM_SIGN)
                    if i != -1:
                        ret += self.buffer[:i]; self.buffer = self.buffer[i:]
                        self.asfh.buffer = self.buffer[:len(common.FRM_SIGN)]
                        self.buffer = self.buffer[len(common.FRM_SIGN):]
                    else:
                        ret += self.buffer[:-len(common.FRM_SIGN) + 1]
                        self.buffer = self.buffer[-len(common.FRM_SIGN) + 1:]
                        break
                header_result, self.buffer = self.asfh.read(self.buffer)
                match header_result:
                    case 'Complete': continue
                    case 'ForceFlush': ret += self.asfh.force_flush();  break
                    case 'Incomplete': break

        return ret

    def flush(self) -> bytes:
        ret = self.buffer
        self.buffer = b''
        return ret
