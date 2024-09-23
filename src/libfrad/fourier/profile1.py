from scipy.fft import dct, idct
import numpy as np
from .prf import compact
from .tools import p1tools
import struct, zlib
from .prf import compact

DEPTHS = (8, 12, 16, 24, 32, 48, 64)

@staticmethod
def analogue(pcm: np.ndarray, bits: int, srate: int, loss_level: float) -> tuple[bytes, int, int, int]:
    # DCT
    pcm = np.pad(pcm, ((0, min((x for x in compact.SAMPLES_LI if x >= len(pcm)), default=len(pcm))-len(pcm)), (0, 0)), mode='constant')
    srate = compact.get_valid_srate(srate)
    dlen, channels = len(pcm), len(pcm[0])
    freqs = np.array([dct(pcm[:, i], norm='forward') for i in range(channels)]) * (2**(bits-1))

    # Quantisation
    mask_freqs = []
    mask_thres = []
    for c in range(channels):
        mapping = p1tools.subband.mapping_to_opus(np.abs(freqs[c]), srate)
        thres = p1tools.subband.mask_thres_mos(mapping, p1tools.spread_alpha) * loss_level
        mask_thres.append(thres * np.sqrt(3)**(16-bits))

        div_factor = p1tools.subband.mapping_from_opus(thres, dlen, srate)
        masked = np.array(p1tools.quant(freqs[c] / div_factor))
        mask_freqs.append(masked)

    freqs, thres = np.array(mask_freqs).round().astype(int), np.array(mask_thres).round().astype(int)

    # Ravelling and packing
    thres_gol = p1tools.exp_golomb_rice_encode(thres.T.ravel())
    freqs_gol = p1tools.exp_golomb_rice_encode(freqs.T.ravel())
    frad = struct.pack(f'>I', len(thres_gol)) + thres_gol + freqs_gol

    # Deflating
    frad = zlib.compress(frad, level=9)

    return frad, DEPTHS.index(bits), channels, srate

@staticmethod
def digital(fradc: bytes, fb: int, channels: int, srate: int, fsize: int) -> np.ndarray:
    bits = DEPTHS[fb]

    # Inflating
    frad = b''
    try: frad = zlib.decompress(fradc)
    except: return np.zeros((fsize, channels))
    thresbytes, frad = struct.unpack(f'>I', frad[:4])[0], frad[4:]
    thres, frad = p1tools.exp_golomb_rice_decode(frad[:thresbytes]).reshape(-1, channels).T.astype(float) / (np.sqrt(3)**(16-bits)), frad[thresbytes:]

    # Unpacking and unravelling
    freqs: np.ndarray = p1tools.exp_golomb_rice_decode(frad).astype(float).reshape(-1, channels).T

    # Removing potential Infinities and Non-numbers
    freqs = np.where(np.isnan(freqs) | np.isinf(freqs), 0, freqs)
    thres = np.where(np.isnan(thres) | np.isinf(thres), 0, thres)

    # Dequantisation
    freqs = np.array([p1tools.dequant(freqs[c]) * p1tools.subband.mapping_from_opus(thres[c], len(freqs[c]), srate) for c in range(channels)])

    # Inverse DCT and stacking
    return np.ascontiguousarray(np.array([idct(chnl, norm='forward') for chnl in freqs]).T) / (2**(bits-1))
