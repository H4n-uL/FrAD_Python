import numpy as np
from libfrad import fourier, common
from libfrad.fourier.prf import profiles
from libfrad.tools import ecc
from libfrad.tools.asfh import ASFH
from libfrad.tools.stream import StreamInfo
import zlib

EMPTY = np.array([]).shape

class Decoder:
    def __init__(self, fix_error: bool = False):
        self.asfh = ASFH()
        self.info = ASFH()
        self.buffer = b''
        self.overlap_fragment = np.array([])
        self.streaminfo = StreamInfo()
        self.fix_error = fix_error

    def overlap(self, frame: np.ndarray) -> np.ndarray:
        if self.overlap_fragment.shape != EMPTY:
            fade = np.linspace(0.0, 1.0, len(self.overlap_fragment))
            for sample, overlap_sample, fade_in, fade_out in zip(frame, self.overlap_fragment, fade, fade[::-1]):
                for s, o in zip(sample, overlap_sample):
                    s = s * fade_in + o * fade_out
        next_overlap = np.array([])
        if self.asfh.profile in profiles.COMPACT and self.asfh.overlap != 0:
            overlap_ratio = self.asfh.overlap
            frame_cutout = len(frame) * (overlap_ratio - 1) // overlap_ratio
            next_overlap = frame[frame_cutout:]
            frame = frame[:frame_cutout]
        self.overlap_fragment = next_overlap
        return frame

    def is_empty(self) -> bool:
        return len(self.buffer) < len(common.FRM_SIGN)

    def process(self, stream: bytes) -> tuple[np.ndarray, int, bool]:
        self.buffer += stream
        ret = []

        while True:
            if self.asfh.all_set:
                if len(self.buffer) < self.asfh.frmbytes: break
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
                    case 4: pcm = fourier.profile4.digital(frad, self.asfh.bit_depth_index, self.asfh.channels, self.asfh.endian)
                    case _: pcm = fourier.profile0.digital(frad, self.asfh.bit_depth_index, self.asfh.channels, self.asfh.endian)

                pcm = self.overlap(pcm)
                samples = len(pcm)

                ret.append(pcm)
                self.asfh.clear()
                self.streaminfo.update(self.asfh.total_bytes, samples, self.asfh.srate)
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
                                ret.extend(self.flush()[0])
                                return np.concatenate(ret), self.asfh.srate, True
                            
                    case 'ForceFlush':
                        ret.extend(self.flush()[0])
                        break

                    case 'Incomplete':
                        break

        if not ret: return np.array([]), self.asfh.srate, False
        return np.concatenate(ret), self.asfh.srate, False

    def flush(self) -> tuple[np.ndarray, int, bool]:
        ret = self.overlap_fragment
        self.streaminfo.update(0, len(self.overlap_fragment), self.asfh.srate)
        self.overlap_fragment = np.array([])
        self.asfh.clear()
        return ret, self.asfh.srate, False
