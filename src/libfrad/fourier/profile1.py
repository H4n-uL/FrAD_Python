from scipy.fft import dct, idct
import numpy as np
from .profiles import compact
from .tools import p1tools
import struct, zlib

DEPTHS = (8, 12, 16, 24, 32, 48, 64)

def get_scale_factors(bits: int) -> tuple[float, float]:
    pcm_factor = 2**(bits-1)
    thres_factor = np.sqrt(3)**(16-bits)
    return pcm_factor, thres_factor

def finite(arr: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(arr) | np.isinf(arr), 0, arr)

def untrim(arr: np.ndarray, fsize: int, channels: int) -> np.ndarray:
    return np.pad(arr, (0, max(0, (fsize*channels)-len(arr))), 'constant')

@staticmethod
def analogue(pcm: np.ndarray, bits: int, srate: int, loss_level: float) -> tuple[bytes, int, int, int]:
    pcm_factor, thres_factor = get_scale_factors(bits)
    # DCT
    pcm = np.pad(pcm, ((0, min((x for x in compact.SAMPLES_LI if x >= len(pcm)), default=len(pcm))-len(pcm)), (0, 0)), mode='constant')
    srate = compact.get_valid_srate(srate)
    dlen, channels = len(pcm), len(pcm[0])
    freqs = np.array([dct(pcm[:, i], norm='forward') for i in range(channels)]) * pcm_factor

    # Quantisation
    freqs_masked = []
    thresholds = []
    for c in range(channels):
        mapping = p1tools.subband.mapping_to_opus(np.abs(freqs[c]), srate)
        thres = p1tools.subband.mask_thres_mos(mapping, p1tools.spread_alpha) * loss_level

        div_factor = p1tools.subband.mapping_from_opus(thres, dlen, srate)
        chnl_masked = np.array(p1tools.quant(freqs[c] / div_factor))

        freqs_masked.append(finite(chnl_masked))
        thresholds.append(finite(thres * thres_factor))

    freqs, thres = np.array(freqs_masked).round().astype(int), np.array(thresholds).round().astype(int)

    # Ravelling and packing
    thres_gol = p1tools.exp_golomb_rice_encode(thres.T.ravel())
    freqs_gol = p1tools.exp_golomb_rice_encode(freqs.T.ravel())
    frad = struct.pack(f'>I', len(thres_gol)) + thres_gol + freqs_gol
    # frad = frad[:(4 + len(thres_gol) + int(len(freqs_gol) * 0.75))]

    # Deflating
    frad = zlib.compress(frad, level=9)

    return frad, DEPTHS.index(bits), channels, srate

@staticmethod
def digital(frad: bytes, fb: int, channels: int, srate: int, fsize: int) -> np.ndarray:
    bits = DEPTHS[fb]
    pcm_factor, thres_factor = get_scale_factors(bits)

    # Inflating
    try: frad = zlib.decompress(frad)
    except: return np.zeros((fsize, channels))
    thresbytes, frad = struct.unpack(f'>I', frad[:4])[0], frad[4:]
    thres_gol, frad = frad[:thresbytes], frad[thresbytes:]

    # Unpacking and unravelling
    thres_flat = untrim(p1tools.exp_golomb_rice_decode(thres_gol).astype(float) / thres_factor, fsize, channels)
    freqs_flat = untrim(p1tools.exp_golomb_rice_decode(frad).astype(float), fsize, channels)

    thresholds = thres_flat.reshape(-1, channels).T
    freqs_masked = freqs_flat.reshape(-1, channels).T

    # Dequantisation
    freqs = np.array([p1tools.dequant(freqs_masked[c]) * p1tools.subband.mapping_from_opus(thresholds[c], fsize, srate) for c in range(channels)])

    # Inverse DCT and stacking
    return np.ascontiguousarray(np.array([idct(chnl, norm='forward') for chnl in freqs]).T) / pcm_factor
