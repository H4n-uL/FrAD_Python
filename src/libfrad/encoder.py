import numpy as np
from libfrad import fourier
from libfrad.fourier import AVAILABLE, SEGMAX, BIT_DEPTHS, profiles
from libfrad.fourier.profiles import compact
from libfrad.tools import ecc
from libfrad.tools.asfh import ASFH
from libfrad.tools.process import ProcessInfo
import sys
from libfrad.backend.pcmformat import ff_format_to_numpy_type
# import random

EMPTY = np.array([]).shape

class Encoder:
    def __init__(self, profile: int, pcm_format: str):
        if profile not in AVAILABLE: print(f"Invalid profile! Available: {AVAILABLE}", file=sys.stderr); exit(1)

        self.asfh = ASFH()
        self.asfh.profile = profile
        self.buffer = b''
        self.bit_depth = 0
        self.channels = 0
        self.fsize = 0
        self.srate = 0
        self.overlap_fragment = np.array([])
        self.procinfo = ProcessInfo()

        self.pcm_format = ff_format_to_numpy_type(pcm_format)
        self.loss_level = 0.5

    def set_channels(self, channels: int):
        if channels == 0: print("Channel count cannot be zero", file=sys.stderr); exit(1)
        self.channels = channels

    def set_frame_size(self, frame_size: int):
        if frame_size == 0: print("Frame size cannot be zero", file=sys.stderr); exit(1)
        if frame_size > SEGMAX[self.asfh.profile]: print(f"Samples per frame cannot exceed {SEGMAX[self.asfh.profile]}", file=sys.stderr); exit(1)
        self.fsize = frame_size

    def set_srate(self, srate: int):
        if srate == 0: print("Sample rate cannot be zero", file=sys.stderr); exit(1)
        if self.asfh.profile in profiles.COMPACT:
            x = compact.get_valid_srate(srate)
            if x != srate:
                print(f"Invalid sample rate! Valid rates for profile {self.asfh.profile}: {compact.SRATES}", file=sys.stderr)
                print(f"Auto-adjusting to: {x}", file=sys.stderr)
            srate = x
        self.srate = srate

    def set_bit_depth(self, bit_depth: int):
        if bit_depth == 0: print("Bit depth cannot be zero", file=sys.stderr); exit(1)
        if bit_depth not in BIT_DEPTHS[self.asfh.profile]: 
            print(f"Invalid bit depth! Valid depths for profile {self.asfh.profile}: {list(filter(lambda x: x != 0, BIT_DEPTHS[self.asfh.profile]))}", file=sys.stderr)
            exit(1)
        self.bit_depth = bit_depth

    def set_ecc(self, ecc: bool, ecc_ratio: tuple[int, int]):
        self.asfh.ecc = ecc
        if ecc_ratio[0] == 0:
            print("ECC data size must not be zero", file=sys.stderr)
            print("Setting ECC to default 96 24", file=sys.stderr)
            ecc_ratio = (96, 24)
        if ecc_ratio[0] + ecc_ratio[1] > 255:
            print(f"ECC data size and check size must not exceed 255, given: {ecc_ratio[0]} and {ecc_ratio[1]}", file=sys.stderr)
            print("Setting ECC to default 96 24", file=sys.stderr)
            ecc_ratio = (96, 24)
        self.asfh.ecc_dsize, self.asfh.ecc_codesize = ecc_ratio

    def set_little_endian(self, little_endian: bool): self.asfh.endian = little_endian
    def set_loss_level(self, loss_level: float): self.loss_level = max(abs(loss_level), 0.125)
    def set_overlap_ratio(self, overlap_ratio: int):
        if overlap_ratio != 0: overlap_ratio = max(2, min(256, overlap_ratio))
        self.asfh.overlap_ratio = overlap_ratio

    def overlap(self, frame: np.ndarray) -> np.ndarray:
        if self.overlap_fragment.shape != EMPTY:
            frame = np.concatenate((self.overlap_fragment, frame), axis=0)
        next_overlap = np.array([])
        if self.asfh.profile in profiles.COMPACT and self.asfh.overlap_ratio > 1:
            frame_cutout = len(frame) * (self.asfh.overlap_ratio - 1) // self.asfh.overlap_ratio
            next_overlap = frame[frame_cutout:]
        self.overlap_fragment = next_overlap
        return frame
    
    def inner(self, stream: bytes, flush: bool) -> bytes:
        self.buffer += stream
        ret = b''

        while True:
            # self.asfh.profile = random.choice(AVAILABLE)
            # self.bit_depth = random.choice(list(filter(lambda x: x != 0, BIT_DEPTHS[self.asfh.profile])))
            # self.set_frame_size(
            #     random.choice(compact.SAMPLES_LI) if self.asfh.profile in profiles.COMPACT
            #     else random.randint(128, 32768)
            # )
            # self.set_loss_level(random.uniform(0.125, 10.0))
            # ecc_data = random.randint(1, 255)
            # self.set_ecc(random.random() < 0.5, (ecc_data, random.randint(0, 255 - ecc_data)))
            # self.set_overlap_ratio(random.randint(2, 256))

            rlen = self.fsize
            if self.asfh.profile in profiles.COMPACT:
                li_val = min(filter(lambda x: x >= self.fsize, compact.SAMPLES_LI))
                if li_val < len(self.overlap_fragment):
                    rlen = min(filter(lambda x: x >= len(self.overlap_fragment), compact.SAMPLES_LI)) - len(self.overlap_fragment)
                else:
                    rlen = li_val - len(self.overlap_fragment)

            bytes_per_sample = self.pcm_format.itemsize
            read_bytes = rlen * self.channels * bytes_per_sample
            if len(self.buffer) < read_bytes and not flush: break

            pcm_bytes, self.buffer = self.buffer[:read_bytes], self.buffer[read_bytes:]
            frame = np.frombuffer(pcm_bytes, self.pcm_format).reshape(-1, self.channels)

            if frame.size == 0: self.asfh.force_flush(); break
            samples = len(frame)

            frame = self.overlap(frame); fsize = len(frame)
            frad, bit_depth_index, channels, srate = None, 0, 0, 0
            match self.asfh.profile:
                case 1: frad, bit_depth_index, channels, srate = fourier.profile1.analogue(frame, self.bit_depth, self.srate, self.loss_level)
                case 4: frad, bit_depth_index, channels, srate = fourier.profile4.analogue(frame, self.bit_depth, self.srate, self.asfh.endian)
                case _: frad, bit_depth_index, channels, srate = fourier.profile0.analogue(frame, self.bit_depth, self.srate, self.asfh.endian)

            if self.asfh.ecc: frad = ecc.encode(frad, self.asfh.ecc_dsize, self.asfh.ecc_codesize)
            self.asfh.bit_depth_index, self.asfh.channels, self.asfh.fsize, self.asfh.srate = bit_depth_index, channels, fsize, srate
            ret += self.asfh.write(frad)
            if flush: self.asfh.force_flush()

            self.procinfo.update(self.asfh.total_bytes, samples, self.asfh.srate)

        return ret
    
    def process(self, stream: bytes) -> bytes: return self.inner(stream, False)
    def flush(self) -> bytes: return self.inner(b'', True)
