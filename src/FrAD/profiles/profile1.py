from scipy.fft import dct, idct
import numpy as np
from .tools import p1tools
import struct, zlib

class p1:
    srates = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000)
    smpls = {128: [128 * 2**i for i in range(8)], 144: [144 * 2**i for i in range(8)], 192: [192 * 2**i for i in range(8)]}
    smpls_li = tuple([item for sublist in smpls.values() for item in sublist])

    depths = (8, 12, 16, 24, 32, 48, 64)
    dtypes = {64:'i8',48:'i8',32:'i4',24:'i4',16:'i2',12:'i2',8:'i1'}

    @staticmethod
    def signext_24x(byte: bytes, bits, be):
        return (int((be and byte.hex()[0] or byte.hex()[-1]), base=16) > 7 and b'\xff' or b'\x00') * (bits//24) + byte

    @staticmethod
    def signext_12(hex_str):
        if len(hex_str)!=3: return ''
        return (int(hex_str[0], base=16) > 7 and 'f' or '0') + hex_str

    @staticmethod
    def analogue(pcm: np.ndarray, bits: int, channels: int, little_endian: bool, kwargs) -> tuple[bytes, int, int, int]:
        be = not little_endian
        endian = be and '>' or '<'

        # DCT
        pcm = np.pad(pcm, ((0, min((x for x in p1.smpls_li if x >= len(pcm)), default=len(pcm))-len(pcm)), (0, 0)), mode='constant')
        dlen = len(pcm)
        freqs = np.array([dct(pcm[:, i]*(2**(bits-1))) for i in range(channels)]) / dlen

        # Quantisation
        freqs, pns = p1tools.quant(freqs, channels, dlen, kwargs)

        # Ravelling and packing
        pns_glm = p1tools.exp_golomb_rice_encode(np.frombuffer(np.array(pns.T/(2**(bits-1))).astype(endian+'e').tobytes(), dtype=f'{endian}i2'))
        frad: bytes = p1tools.exp_golomb_rice_encode(freqs.T.ravel().astype(int))
        frad = struct.pack(f'{endian}I', len(pns_glm)) + pns_glm + frad

        # Deflating
        frad = zlib.compress(frad, level=9)

        return frad, bits, channels, p1.depths.index(bits)

    @staticmethod
    def digital(frad: bytes, fb: int, channels: int, little_endian: bool, kwargs) -> np.ndarray:
        be = not little_endian
        endian = be and '>' or '<'
        bits = p1.depths[fb]

        # Inflating
        frad = zlib.decompress(frad)
        thresbytes, frad = struct.unpack(f'{endian}I', frad[:4])[0], frad[4:]
        thres_int, frad = p1tools.exp_golomb_rice_decode(frad[:thresbytes]).astype(f'{endian}i2').tobytes(), frad[thresbytes:]
        thres = np.frombuffer(thres_int, dtype=f'{endian}f2').reshape((-1, channels)).T * (2**(bits-1))

        # Unpacking and unravelling
        freqs: np.ndarray = p1tools.exp_golomb_rice_decode(frad).astype(float).reshape(-1, channels).T

        # Removing potential Infinities and Non-numbers
        freqs = np.where(np.isnan(freqs) | np.isinf(freqs), 0, freqs)

        # Dequantisation
        freqs = p1tools.dequant(freqs, channels, thres, kwargs)

        # Inverse DCT and stacking
        return np.ascontiguousarray(np.array([idct(chnl*len(chnl)) for chnl in freqs]).T)/(2**(bits-1))
