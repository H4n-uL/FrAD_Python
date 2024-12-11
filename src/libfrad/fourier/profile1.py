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

def untrim(arr: np.ndarray, fsize: int, channels: int) -> np.ndarray:
    return np.pad(arr, (0, max(0, (fsize*channels)-len(arr))), 'constant')

def analogue(pcm: np.ndarray, bits: int, srate: int, loss_level: float) -> tuple[bytes, int, int, int]:
    if bits not in DEPTHS: bits = 16
    pcm_factor, thres_factor = get_scale_factors(bits)
    # DCT
    pcm = np.pad(pcm, ((0, min((x for x in compact.SAMPLES_LI if x >= len(pcm)), default=len(pcm))-len(pcm)), (0, 0)), mode='constant')
    srate, loss_level, dlen, channels = compact.get_valid_srate(srate), max(abs(loss_level), 0.125), len(pcm), len(pcm[0])
    freqs = np.array([dct(pcm[:, i], norm='forward') for i in range(channels)]) * pcm_factor

    # Quantisation
    freqs_masked = []
    thresholds = []
    for c in range(channels):
        thres_channel = p1tools.mask_thres_mos(freqs[c], srate, bits, loss_level, p1tools.SPREAD_ALPHA)
        div_factor = p1tools.mapping_from_opus(thres_channel, dlen, srate)
        div_factor = np.where(div_factor == 0, np.inf, div_factor)

        freqs_masked.append(freqs[c] / div_factor)
        thresholds.append(thres_channel)

    freqs_flat = p1tools.quant(np.array(freqs_masked)).round().astype(int).T.ravel()
    thres_flat = p1tools.quant(np.array(thresholds) * thres_factor).round().astype(int).T.ravel()

    # Ravelling and packing
    thres_gol = p1tools.exp_golomb_rice_encode(thres_flat)
    freqs_gol = p1tools.exp_golomb_rice_encode(freqs_flat)
    frad = struct.pack(f'>I', len(thres_gol)) + thres_gol + freqs_gol
    # frad = frad[:(4 + len(thres_gol) + int(len(freqs_gol) * 0.75))]

    # Deflating
    frad = zlib.compress(frad, level=9)

    return frad, DEPTHS.index(bits), channels, srate

def digital(frad: bytes, fb: int, channels: int, srate: int, fsize: int) -> np.ndarray:
    bits = DEPTHS[fb]
    pcm_factor, thres_factor = get_scale_factors(bits)

    # Inflating
    try: frad = zlib.decompress(frad)
    except: return np.zeros((fsize, channels))
    thresbytes, frad = struct.unpack(f'>I', frad[:4])[0], frad[4:]
    thres_gol, frad = frad[:thresbytes], frad[thresbytes:]

    # Unpacking and unravelling
    freqs_flat = p1tools.dequant(p1tools.exp_golomb_rice_decode(frad).astype(float))
    thres_flat = p1tools.dequant(p1tools.exp_golomb_rice_decode(thres_gol).astype(float)) / thres_factor
    freqs_flat = untrim(freqs_flat, fsize, channels)
    thres_flat = untrim(thres_flat, fsize, channels)

    thresholds = thres_flat.reshape(-1, channels).T
    freqs_masked = freqs_flat.reshape(-1, channels).T

    # Dequantisation
    freqs = np.array([freqs_masked[c] * p1tools.mapping_from_opus(thresholds[c], fsize, srate) for c in range(channels)])

    # Inverse DCT and stacking
    return np.ascontiguousarray(np.array([idct(chnl, norm='forward') for chnl in freqs]).T) / pcm_factor
