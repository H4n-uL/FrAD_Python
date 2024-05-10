from scipy.fft import dct, idct
import numpy as np
from .profiles import profile1, profile2

class fourier:
    depths = [12, 16, 24, 32, 48, 64, 128]
    dtypes = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2',12:'f2'}

    @staticmethod
    def analogue(pcm: np.ndarray, bits: int, channels: int, little_endian: bool, *, profile: int = 0, **kwargs) -> tuple[bytes, int, int, int]:
        if profile == 1: return profile1.analogue(pcm, bits, channels, little_endian, kwargs)
        # if profile == 2: return profile2.analogue(pcm, bits, channels, little_endian, kwargs)

        be = not little_endian
        endian = be and '>' or '<'

        # DCT
        freqs = np.array([dct(pcm[:, i]) for i in range(channels)])

        # Overflow check & Increasing bit depth
        while np.max(np.abs(freqs)) > np.finfo(fourier.dtypes[bits]).max:
            if bits == 128: raise Exception('Overflow with reaching the max bit depth.')
            bits = {12:16, 16:24, 24:32, 32:48, 48:64, 64:128}.get(bits, 128)

        # Ravelling and packing
        frad: bytes = freqs.T.ravel().astype(endian+fourier.dtypes[bits]).tobytes()

        # Cutting off bits
        if bits in [128, 64, 32, 16]:
            pass
        elif bits in [48, 24]:
            frad = b''.join([be and frad[i:i+(bits//8)] or frad[i+(bits//24):i+(bits//6)] for i in range(0, len(frad), bits//6)])
        elif bits == 12:
            hexa = frad.hex()
            frad = bytes.fromhex(''.join([be and hexa[i:i+3] or hexa[i:i+4][0] + hexa[i:i+4][2:] for i in range(0, len(hexa), 4)]))
        else: raise Exception('Illegal bits value.')

        return frad, bits, channels, fourier.depths.index(bits)

    @staticmethod
    def digital(frad: bytes, fb: int, channels: int, little_endian: bool, *, profile: int = 0, **kwargs) -> np.ndarray:
        if profile == 1: return profile1.digital(frad, fb, channels, little_endian, kwargs)
        # if profile == 2: return profile2.digital(frad, bits, channels, little_endian, kwargs)

        be = not little_endian
        endian = be and '>' or '<'
        bits = fourier.depths[fb]

        # Padding bits
        if bits % 3 != 0: pass
        elif bits in [24, 48]:
            frad = b''.join([be and frad[i:i+(bits//8)]+(b'\x00'*(bits//24)) or (b'\x00'*(bits//24))+frad[i:i+(bits//8)] for i in range(0, len(frad), bits//8)])
        elif bits == 12:
            hexa = frad.hex()
            frad = bytes.fromhex(''.join([be and (hexa[i:i+3] + '0') or (hexa[i:i+3][0] + '0' + hexa[i:i+3][1:]) for i in range(0, len(hexa), 3)]))
        else:
            raise Exception('Illegal bits value.')

        # Unpacking and unravelling
        freqs: np.ndarray = np.frombuffer(frad, dtype=endian+fourier.dtypes[bits]).astype(float).reshape(-1, channels).T

        # Removing potential Infinities and Non-numbers
        freqs = np.where(np.isnan(freqs) | np.isinf(freqs), 0, freqs)

        # Inverse DCT and stacking
        return np.ascontiguousarray(np.array([idct(chnl) for chnl in freqs]).T)
