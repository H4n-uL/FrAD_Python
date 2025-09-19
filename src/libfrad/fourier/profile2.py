from scipy.fft import dct, idct
import numpy as np
from .profiles import compact
from .tools import p1tools, p2tools
import struct, zlib

DEPTHS = (8, 10, 12, 14, 16, 20, 24)

def get_scale_factors(bits: int) -> float:
    return 2.0 ** (bits - 1)

def untrim(arr: np.ndarray, fsize: int, channels: int) -> np.ndarray:
    return np.pad(arr, (0, max(0, (fsize*channels)-len(arr))), 'constant')

def analogue(pcm: np.ndarray, bits: int, srate: int, loss_level: float) -> tuple[bytes, int, int, int]:
    if bits not in DEPTHS: bits = 16
    pcm_factor = get_scale_factors(bits)
    # DCT
    pcm = np.pad(pcm, ((0, compact.get_samples_min_ge(len(pcm))-len(pcm)), (0, 0)), mode='constant')
    srate, loss_level, dlen, channels = compact.get_valid_srate(srate), max(abs(loss_level), 0.125), len(pcm), len(pcm[0])
    freqs = np.array([dct(pcm[:, i], norm='forward') for i in range(channels)])

    # Quantisation
    freqs_masked = []
    thresholds = []
    lpc_quant = []
    for c in range(channels):
        thres_channel = p1tools.mask_thres_mos(freqs[c] * pcm_factor, srate, loss_level, p1tools.SPREAD_ALPHA)
        div_factor = p1tools.mapping_from_opus(thres_channel, dlen, srate)
        div_factor = np.where(div_factor == 0, np.inf, div_factor)

        tns_freqs, lpc_chnl = p2tools.tns_analysis(freqs[c] / div_factor)
        freqs_masked.append(tns_freqs)
        thresholds.append(thres_channel)
        lpc_quant.append(lpc_chnl)

    freqs_flat = p1tools.quant(
        np.array(freqs_masked) * pcm_factor
    ).round().astype(int).T.ravel()

    thres_flat = p1tools.dequant(
        np.log(np.array(thresholds).clip(min=1.0)) / np.log(np.e / 2)
    ).round().astype(int).T.ravel()

    lpc_flat = np.array(lpc_quant).astype(int).T.ravel()

    # Ravelling and packing
    lpc_gol = p1tools.exp_golomb_rice_encode(lpc_flat)
    thres_gol = p1tools.exp_golomb_rice_encode(thres_flat)
    freqs_gol = p1tools.exp_golomb_rice_encode(freqs_flat)
    frad = struct.pack(f'>H', len(lpc_gol)) + lpc_gol + struct.pack(f'>I', len(thres_gol)) + thres_gol + freqs_gol

    # Deflating
    frad = zlib.compress(frad, wbits=-15)

    return frad, DEPTHS.index(bits), channels, srate

def digital(frad: bytes, fb: int, channels: int, srate: int, fsize: int) -> np.ndarray:
    bits = DEPTHS[fb]
    pcm_factor = get_scale_factors(bits)

    # Inflating
    try: frad = zlib.decompress(frad, wbits=-15)
    except: return np.zeros((fsize, channels))
    lpc_len, frad = struct.unpack(f'>H', frad[:2])[0], frad[2:]
    lpc_gol, frad = frad[:lpc_len], frad[lpc_len:]
    thres_len, frad = struct.unpack(f'>I', frad[:4])[0], frad[4:]
    thres_gol, frad = frad[:thres_len], frad[thres_len:]

    # Unpacking and unravelling
    freqs_flat = p1tools.dequant(p1tools.exp_golomb_rice_decode(frad).astype(float)) / pcm_factor
    thres_flat = np.power(np.e / 2, p1tools.quant(p1tools.exp_golomb_rice_decode(thres_gol).astype(float)))
    lpc_flat =  p1tools.exp_golomb_rice_decode(lpc_gol)
    freqs_flat = untrim(freqs_flat, fsize, channels)
    thres_flat = untrim(thres_flat, p1tools.SUBBANDS, channels)
    lpc_flat =  untrim(lpc_flat, p2tools.MAX_ORDER + 1, channels)

    freqs_masked = freqs_flat.reshape(-1, channels).T
    thresholds = thres_flat.reshape(-1, channels).T
    lpc_quant = lpc_flat.reshape(-1, channels).T

    # Dequantisation
    freqs = np.array(
        [
            p2tools.tns_synthesis(freqs_masked[c], lpc_quant[c]) *
            p1tools.mapping_from_opus(thresholds[c], fsize, srate) for c in range(channels)
        ]
    )

    # Inverse DCT and stacking
    return np.ascontiguousarray(np.array([idct(chnl, norm='forward') for chnl in freqs]).T)
