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
    alpha = 0.8

    @staticmethod
    def signext_24x(byte: bytes, bits, be):
        return (int((be and byte.hex()[0] or byte.hex()[-1]), base=16) > 7 and b'\xff' or b'\x00') * (bits//24) + byte

    @staticmethod
    def signext_12(hex_str):
        if len(hex_str)!=3: return ''
        return (int(hex_str[0], base=16) > 7 and 'f' or '0') + hex_str

    @staticmethod
    def analogue(pcm: np.ndarray, bits: int, channels: int, **kwargs) -> tuple[bytes, int, int]:
        # DCT
        pcm = np.pad(pcm, ((0, min((x for x in p1.smpls_li if x >= len(pcm)), default=len(pcm))-len(pcm)), (0, 0)), mode='constant')
        dlen = len(pcm)
        freqs = np.array([dct(pcm[:, i], norm='forward') for i in range(channels)]) * (2**(bits-1))

        const_factor = 1.25**kwargs['level'] / 19 + 0.5

        # Quantisation
        mask_freqs = []
        mask_thres = []
        for c in range(channels):
            mapping = p1tools.subband.mapping2opus(np.abs(freqs[c]), kwargs['srate'])
            thres = p1tools.subband.mask_thres_MOS(mapping, p1.alpha) * const_factor
            mask_thres.append(thres * 2**(16-bits))
            div_factor = p1tools.subband.mappingfromopus(thres,dlen, kwargs['srate'])

            masked = np.array(np.around(p1tools.quant(freqs[c] / div_factor)))
            mask_freqs.append(masked.astype(int))

        freqs, thres = np.array(mask_freqs), np.array(mask_thres).astype(int)

        # Ravelling and packing
        thres_gol = p1tools.exp_golomb_rice_encode(thres.T.ravel())
        freqs_gol = p1tools.exp_golomb_rice_encode(freqs.T.ravel())
        frad = struct.pack(f'>I', len(thres_gol)) + thres_gol + freqs_gol

        # Deflating
        frad = zlib.compress(frad, level=9)

        return frad, p1.depths.index(bits), channels

    @staticmethod
    def digital(frad: bytes, fb: int, channels: int, **kwargs) -> np.ndarray:
        bits = p1.depths[fb]

        # Inflating
        frad = zlib.decompress(frad)
        thresbytes, frad = struct.unpack(f'>I', frad[:4])[0], frad[4:]
        thres, frad = p1tools.exp_golomb_rice_decode(frad[:thresbytes]).reshape(-1, channels).T.astype(float) / (2**(16-bits)), frad[thresbytes:]

        # Unpacking and unravelling
        freqs: np.ndarray = p1tools.exp_golomb_rice_decode(frad).astype(float).reshape(-1, channels).T

        # Removing potential Infinities and Non-numbers
        freqs = np.where(np.isnan(freqs) | np.isinf(freqs), 0, freqs)
        thres = np.where(np.isnan(thres) | np.isinf(thres), 0, thres)

        # Dequantisation
        freqs = np.array([p1tools.dequant(freqs[c]) * p1tools.subband.mappingfromopus(thres[c], len(freqs[c]), kwargs['srate']) for c in range(channels)])

        # Inverse DCT and stacking
        return np.ascontiguousarray(np.array([idct(chnl, norm='forward') for chnl in freqs]).T) / (2**(bits-1))
