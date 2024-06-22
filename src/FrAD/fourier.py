from scipy.fft import dct, idct
import numpy as np
from .profiles.profile1 import p1

class fourier:
    depths = (12, 16, 24, 32, 48, 64, 128)
    dtypes = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2',12:'f2'}
    float_dr = {12: 5, 16: 5, 24: 8, 32: 8, 48: 11, 64: 11, 128: 15}

    @staticmethod
    def analogue(pcm: np.ndarray, bits: int, channels: int, little_endian: bool, *, profile: int = 0, **kwargs) -> tuple[bytes, int, int, int]:
        if profile == 1: return p1.analogue(pcm, bits, channels, little_endian, kwargs)

        be = not little_endian
        endian = be and '>' or '<'

        # DCT
        freqs = np.array([dct(pcm[:, i]) for i in range(channels)])

        # Overflow check & Increasing bit depth
        while np.max(np.abs(freqs)) > 2**(2**fourier.float_dr[bits]):
            if bits == 128: raise Exception('Overflow with reaching the max bit depth.')
            bits = {12:16, 16:24, 24:32, 32:48, 48:64, 64:128}.get(bits, 128)

        # Ravelling and packing
        if bits%8!=0: endian = '>'
        frad: bytes = freqs.T.ravel().astype(endian+fourier.dtypes[bits]).tobytes()

        # Cutting off bits
        if bits in (128, 64, 32, 16):
            pass
        elif bits in (48, 24):
            frad = b''.join([be and frad[i:i+(bits//8)] or frad[i+(bits//24):i+(bits//6)] for i in range(0, len(frad), bits//6)])
        elif bits == 12:
            hexa = frad.hex()
            hexa = ''.join([hexa[i:i+3] for i in range(0, len(hexa), 4)])
            if len(hexa)%2!=0: hexa+='0'
            frad = bytes.fromhex(hexa)
        else: raise Exception('Illegal bits value.')

        return frad, bits, channels, fourier.depths.index(bits)

    @staticmethod
    def digital(frad: bytes, fb: int, channels: int, little_endian: bool, *, profile: int = 0, **kwargs) -> np.ndarray:
        if profile == 1: return p1.digital(frad, fb, channels, little_endian, kwargs)

        be = not little_endian
        endian = be and '>' or '<'
        bits = fourier.depths[fb]

        # Padding bits
        if bits % 3 != 0: pass
        elif bits in (24, 48):
            frad = b''.join([be and frad[i:i+(bits//8)]+(b'\x00'*(bits//24)) or (b'\x00'*(bits//24))+frad[i:i+(bits//8)] for i in range(0, len(frad), bits//8)])
        elif bits == 12:
            hexa = frad.hex()
            if len(hexa)%3!=0: hexa=hexa[:-1]
            frad = bytes.fromhex(''.join([f'{hexa[i:i+3]}0' for i in range(0, len(hexa), 3)]))
        else: raise Exception('Illegal bits value.')

        # Unpacking and unravelling
        if bits%8!=0: endian = '>'
        freqs: np.ndarray = np.frombuffer(frad, endian+fourier.dtypes[bits]).astype(float).reshape(-1, channels).T

        # Removing potential Infinities and Non-numbers
        freqs = np.where(np.isnan(freqs) | np.isinf(freqs), 0, freqs)

        # Inverse DCT and stacking
        return np.ascontiguousarray(np.array([idct(chnl) for chnl in freqs]).T)
