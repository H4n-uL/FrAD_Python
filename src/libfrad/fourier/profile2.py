from scipy.fft import dct, idct
import numpy as np
from .profiles import compact
from .tools import p1tools, p2tools
import struct, zlib

DEPTHS = (8, 9, 10, 11, 12, 14, 16)

@staticmethod
def analogue(pcm: np.ndarray, bits: int, srate: int) -> tuple[bytes, int, int, int]:
    # DCT
    pcm = np.pad(pcm, ((0, min((x for x in compact.SAMPLES_LI if x >= len(pcm)), default=len(pcm))-len(pcm)), (0, 0)), mode='constant')
    srate, channels = compact.get_valid_srate(srate), len(pcm[0])
    freqs = np.array([dct(pcm[:, i], norm='forward') for i in range(channels)]) * (2**(bits-1))

    # Quantisation
    tns_freqs, lpc = p2tools.tns.analysis(freqs)

    # Ravelling and packing
    lpc_bytes = p1tools.exp_golomb_rice_encode(lpc.ravel().astype(int))
    frad: bytes = p1tools.exp_golomb_rice_encode(tns_freqs.ravel().astype(int))
    frad = struct.pack(f'>I', len(lpc_bytes)) + lpc_bytes + frad

    # Deflating
    frad = zlib.compress(frad)

    return frad, DEPTHS.index(bits), channels, srate

@staticmethod
def digital(frad: bytes, fb: int, channels: int, srate: int, fsize: int) -> np.ndarray:
    bits = DEPTHS[fb]

    # Inflating
    frad = zlib.decompress(frad)
    lpclen, frad = struct.unpack(f'>I', frad[:4])[0], frad[4:]
    lpc, frad = p1tools.exp_golomb_rice_decode(frad[:lpclen]).reshape(channels, -1), frad[lpclen:]

    # Unpacking
    freqs: np.ndarray = p1tools.exp_golomb_rice_decode(frad).astype(float).reshape(channels, -1)

    # Removing potential Infinities and Non-numbers
    freqs = np.where(np.isnan(freqs) | np.isinf(freqs), 0, freqs)

    # Dequantisation
    rev_freqs = p2tools.tns.synthesis(freqs, lpc)

    # Inverse DCT and stacking
    return np.ascontiguousarray(np.array([idct(chnl, norm='forward') for chnl in rev_freqs]).T) / (2**(bits-1))
