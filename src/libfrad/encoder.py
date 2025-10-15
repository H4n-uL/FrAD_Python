import numpy as np
from . import fourier
from .backend.pcmformat import ff_format_to_numpy_type, to_f64
from .fourier import AVAILABLE, SEGMAX, BIT_DEPTHS, profiles
from .fourier.profiles import compact
from .tools import ecc
from .tools.asfh import ASFH
import sys
import random

EMPTY = np.array([]).shape

class EncodeResult:
    def __init__(self, buf: bytes, samples: int):
        self.buf = buf
        self.samples = samples

class Encoder:
    def __init__(self, profile: int, srate: int, channels: int, bit_depth: int, frame_size: int, pcm_format: str):
        if profile not in AVAILABLE: print(f"Invalid profile! Available: {AVAILABLE}", file=sys.stderr); exit(1)

        self.asfh = ASFH()
        self.buffer = b''
        self.bit_depth = 0
        self.channels = 0
        self.fsize = 0
        self.srate = 0
        self.overlap_fragment = np.array([])
        self.pcm_format = ff_format_to_numpy_type(pcm_format)
        self.loss_level = 0.5
        self.init = False

        self.set_profile(profile, srate, channels, bit_depth, frame_size)

    def overlap(self, frame: np.ndarray, flush: bool) -> np.ndarray:
        if self.overlap_fragment.shape != EMPTY:
            frame = np.concatenate((self.overlap_fragment, frame), axis=0)

        next_overlap = np.array([])
        next_flag = (
            not flush and
            self.asfh.profile in profiles.COMPACT and
            self.asfh.overlap_ratio > 1
        )
        if next_flag:
            frame_cutout = len(frame) * (self.asfh.overlap_ratio - 1) // self.asfh.overlap_ratio
            next_overlap = frame[frame_cutout:]
        self.overlap_fragment = next_overlap
        return frame

    def inner(self, stream: bytes, flush: bool) -> EncodeResult:
        self.buffer += stream
        ret, samples = b'', 0

        if not self.init:
            return EncodeResult(ret, samples)

        while True:
            # rng = random.Random()
            # prf = rng.choice(AVAILABLE)
            # self.set_profile(prf, self.srate, self.channels,
            #     rng.choice(list(filter(lambda x: x != 0, BIT_DEPTHS[prf]))),
            #     rng.choice(compact.SAMPLES) if prf in profiles.COMPACT else rng.randint(128, 32768)
            # )
            # self.set_loss_level(rng.uniform(0.125, 10.0))
            # ecc_data = rng.randint(1, 255)
            # self.set_ecc(rng.random() < 0.5, (ecc_data, rng.randint(0, 255 - ecc_data)))
            # self.set_overlap_ratio(rng.randint(2, 256))

            overlap_len = len(self.overlap_fragment)
            rlen = max(self.fsize, overlap_len)
            if self.asfh.profile in profiles.COMPACT:
                rlen = compact.get_samples_min_ge(rlen)
            rlen -= overlap_len

            bytes_per_sample = self.pcm_format.itemsize
            read_bytes = rlen * self.channels * bytes_per_sample
            if len(self.buffer) < read_bytes and not flush: break

            pcm_bytes, self.buffer = self.buffer[:read_bytes], self.buffer[read_bytes:]
            frame = np.frombuffer(pcm_bytes, self.pcm_format).reshape(-1, self.channels)
            frame = to_f64(frame, self.pcm_format)
            samples_in_frame = len(frame)

            frame = self.overlap(frame, flush)
            if frame.size == 0 and self.overlap_fragment.shape == EMPTY:
                ret += self.asfh.force_flush()
                break
            samples += samples_in_frame
            fsize = len(frame)

            frad, bit_depth_index, channels, srate = None, 0, 0, 0
            match self.asfh.profile:
                case 1: frad, bit_depth_index, channels, srate = fourier.profile1.analogue(frame, self.bit_depth, self.srate, self.loss_level)
                case 2: frad, bit_depth_index, channels, srate = fourier.profile2.analogue(frame, self.bit_depth, self.srate, self.loss_level)
                case 4: frad, bit_depth_index, channels, srate = fourier.profile4.analogue(frame, self.bit_depth, self.srate, self.asfh.endian)
                case _: frad, bit_depth_index, channels, srate = fourier.profile0.analogue(frame, self.bit_depth, self.srate, self.asfh.endian)

            if self.asfh.ecc: frad = ecc.encode(frad, self.asfh.ecc_dsize, self.asfh.ecc_codesize)
            self.asfh.bit_depth_index, self.asfh.channels, self.asfh.fsize, self.asfh.srate = bit_depth_index, channels, fsize, srate
            ret += self.asfh.write(frad)
            if flush: ret += self.asfh.force_flush()

        return EncodeResult(ret, samples)

    def process(self, stream: bytes) -> EncodeResult: return self.inner(stream, False)
    def flush(self) -> EncodeResult:
        if self.init: return self.inner(b'', True)
        return EncodeResult(b'', 0)

    # Getters and Setters
    @staticmethod
    def verify_profile(profile: int) -> str | None:
        if profile not in AVAILABLE:
            return f"Invalid profile! Available: {AVAILABLE}"
        return None

    @staticmethod
    def verify_srate(profile: int, srate: int) -> str | None:
        if srate == 0:
            return "Sample rate cannot be zero"
        if profile in profiles.COMPACT:
            x = compact.get_valid_srate(srate)
            if x != srate:
                return f"Invalid sample rate! Valid rates for profile {profile}: {compact.SRATES}"
        return None

    @staticmethod
    def verify_channels(profile: int, channels: int) -> str | None:
        if channels == 0:
            return "Channel count cannot be zero"
        return None

    @staticmethod
    def verify_bit_depth(profile: int, bit_depth: int) -> str | None:
        if bit_depth == 0:
            return "Bit depth cannot be zero"
        if bit_depth not in BIT_DEPTHS[profile]:
            return f"Invalid bit depth! Valid depths for profile {profile}: {list(filter(lambda x: x != 0, BIT_DEPTHS[profile]))}"
        return None

    @staticmethod
    def verify_frame_size(profile: int, frame_size: int) -> str | None:
        if frame_size == 0:
            return "Frame size cannot be zero"
        if frame_size > SEGMAX[profile]:
            return f"Samples per frame cannot exceed {SEGMAX[profile]}"
        return None

    def get_profile(self) -> int: return self.asfh.profile
    def set_profile(self, profile: int, srate: int, channels: int, bit_depth: int, frame_size: int) -> str | EncodeResult:
        if (e := self.verify_profile(profile)) is not None: return e
        if (e := self.verify_srate(profile, srate)) is not None: return e
        if (e := self.verify_channels(profile, channels)) is not None: return e
        if (e := self.verify_bit_depth(profile, bit_depth)) is not None: return e
        if (e := self.verify_frame_size(profile, frame_size)) is not None: return e

        res = EncodeResult(b'', 0)
        if (
            self.channels != 0 and self.channels != channels
            or self.srate != 0 and self.srate != srate
        ): res = self.flush()
        self.asfh.profile = profile
        self.srate = srate
        self.channels = channels
        self.bit_depth = bit_depth
        self.fsize = frame_size
        self.init = True

        return res

    def get_channels(self) -> int: return self.channels
    def set_channels(self, channels: int) -> str | EncodeResult:
        self.verify_channels(self.get_profile(), channels)
        res = EncodeResult(b'', 0)
        if self.channels != 0 and self.channels != channels: res = self.flush()
        self.channels = channels
        return res

    def get_srate(self) -> int: return self.srate
    def set_srate(self, srate: int) -> str | EncodeResult:
        if e := self.verify_srate(self.get_profile(), srate): return e
        res = EncodeResult(b'', 0)
        if self.srate != 0 and self.srate != srate: res = self.flush()
        self.srate = srate
        return res

    def get_frame_size(self) -> int: return self.fsize
    def set_frame_size(self, frame_size: int) -> str | None:
        if e := self.verify_frame_size(self.get_profile(), frame_size): return e
        self.fsize = frame_size

    def get_bit_depth(self) -> int: return self.bit_depth
    def set_bit_depth(self, bit_depth: int) -> str | None:
        if e := self.verify_bit_depth(self.get_profile(), bit_depth): return e
        self.bit_depth = bit_depth

    def set_ecc(self, ecc: bool, ecc_ratio: tuple[int, int]):
        self.asfh.ecc = ecc
        dsize_zero, exceed_255 = ecc_ratio[0] == 0, ecc_ratio[0] + ecc_ratio[1] > 255
        if dsize_zero or exceed_255:
            if dsize_zero: print("ECC data size must not be zero", file=sys.stderr)
            if exceed_255: print(f"ECC data size and check size must not exceed 255, given: {ecc_ratio[0]} and {ecc_ratio[1]}", file=sys.stderr)
            print("Setting ECC to default 96 24", file=sys.stderr)
            ecc_ratio = (96, 24)
        self.asfh.ecc_dsize, self.asfh.ecc_codesize = ecc_ratio

    def set_little_endian(self, little_endian: bool): self.asfh.endian = little_endian
    def set_loss_level(self, loss_level: float): self.loss_level = max(abs(loss_level), 0.125)
    def set_overlap_ratio(self, overlap_ratio: int):
        if overlap_ratio != 0: overlap_ratio = max(2, min(256, overlap_ratio))
        self.asfh.overlap_ratio = overlap_ratio
