import numpy as np
from . import fourier, common
from .backend import hanning_in_overlap
from .fourier import profiles
from .tools import ecc
from .tools.asfh import ASFH
import zlib

EMPTY = np.array([]).shape

class DecodeResult:
    def __init__(self, pcm: list[np.ndarray], srate: int, frames: int, crit: bool):
        self.pcm = np.concatenate(pcm) if pcm else np.array([])
        self.srate = srate
        self.frames = frames
        self.crit = crit

class Decoder:
    def __init__(self, fix_error: bool = False):
        self.asfh = ASFH()
        self.info = ASFH()
        self.buffer = b''
        self.overlap_fragment = np.array([])
        self.overlap_prog = 0
        self.fix_error = fix_error
        self.broken_frame = False

    def overlap(self, frame: np.ndarray) -> np.ndarray:
        olap_len = len(self.overlap_fragment)
        if self.overlap_fragment.shape != EMPTY:
            fade_in = hanning_in_overlap(olap_len)
            ov_left = min(olap_len - self.overlap_prog, len(frame))
            for i in range(ov_left):
                i_ov = i + self.overlap_prog
                for j in range(self.asfh.channels):
                    frame[i, j] *= fade_in[i_ov]
                    frame[i, j] += self.overlap_fragment[i_ov, j] * fade_in[olap_len - i_ov - 1]
            self.overlap_prog += ov_left

        if olap_len <= self.overlap_prog:
            self.overlap_fragment = np.array([])
            self.overlap_prog = 0
            if self.asfh.profile in profiles.COMPACT and self.asfh.overlap_ratio != 0:
                frame_cutout = len(frame) * (self.asfh.overlap_ratio - 1) // self.asfh.overlap_ratio
                self.overlap_fragment, frame = frame[frame_cutout:], frame[:frame_cutout]
        return frame

    def is_empty(self) -> bool: return len(self.buffer) < len(common.FRM_SIGN) or self.broken_frame
    def get_asfh(self) -> ASFH: return self.asfh

    def process(self, stream: bytes) -> DecodeResult:
        self.buffer += stream
        ret_pcm, frames = [], 0

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
                pcm = None
                match self.asfh.profile:
                    case 1: pcm = fourier.profile1.digital(frad, self.asfh.bit_depth_index, self.asfh.channels, self.asfh.srate, self.asfh.fsize)
                    case 2: pcm = fourier.profile2.digital(frad, self.asfh.bit_depth_index, self.asfh.channels, self.asfh.srate, self.asfh.fsize)
                    case 4: pcm = fourier.profile4.digital(frad, self.asfh.bit_depth_index, self.asfh.channels, self.asfh.endian)
                    case _: pcm = fourier.profile0.digital(frad, self.asfh.bit_depth_index, self.asfh.channels, self.asfh.endian)

                pcm = self.overlap(pcm)

                ret_pcm.append(pcm)
                frames += 1
                self.asfh.clear()
            else:
                if not self.asfh.buffer[:len(common.FRM_SIGN)] == common.FRM_SIGN:
                    i = self.buffer.find(common.FRM_SIGN)
                    if i != -1:
                        self.buffer = self.buffer[i:]
                        self.asfh.buffer = self.buffer[:len(common.FRM_SIGN)]
                        self.buffer = self.buffer[len(common.FRM_SIGN):]
                    else:
                        self.buffer = self.buffer[-len(common.FRM_SIGN) + 1:]
                        break
                header_result, self.buffer = self.asfh.read(self.buffer)
                match header_result:
                    case 'Complete':
                        if not self.asfh.criteq(self.info):
                            srate, chnl = self.info.srate, self.info.channels
                            self.info = self.asfh
                            if srate or chnl:
                                ret_pcm.append(self.flush().pcm)
                                return DecodeResult(ret_pcm, srate, frames, True)

                    case 'ForceFlush':
                        ret_pcm.append(self.flush().pcm)
                        break

                    case 'Incomplete':
                        break

        return DecodeResult(ret_pcm, self.asfh.srate, frames, False)

    def flush(self) -> DecodeResult:
        ret = self.overlap_fragment
        self.overlap_fragment = np.array([])
        self.asfh.clear()
        return DecodeResult([ret], self.asfh.srate, 0, False)
