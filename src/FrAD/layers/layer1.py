from scipy.fft import dct, idct
import numpy as np
from .tools import layer1 as l1tools
import zlib

dtypes = {128:'i16',64:'i8',48:'i8',32:'i4',24:'i4',16:'i2',12:'i2'}
get_range = lambda fs, sr, x: x is not np.inf and int(fs*x*2/sr+0.5) or 2**32
subband =  [0,      20,     99,     180,    264,    352,    447,    549,
            660,    783,    920,    1072,   1243,   1435,   1651,   1897,
            2174,   2490,   2848,   3255,   3718,   4245,   4844,   5528,
            6306,   7193,   8204,   9355,   10668,  12164,  13869,  15813,
            18029,  20555,  23435,  26718,  30460,  34727,  39590,  45135,
            51456,  58663,  66878,  76244,  86921,  99094,  112971, 128791,
            146827, 167389, 190830, 217553, 248019, 282752, 322348, 367489,
            418951, 477621, 544506, 620758, 707688, 806791, 919773, 1048576, np.inf]
qfactors = [4.03,   4.15,   4.39,   4.51,   4.75,   4.87,   4.99,   5.11,
            5.23,   5.47,   5.71,   5.95,   6.19,   6.31,   6.56,   6.81,
            7.06,   7.08,   7.03,   6.99,   7.05,   7.08,   7.04,   6.54,
            6.45,   6.36,   6.27,   6.17,   6.09,   6.00,   5.41,   4.01,
            3.87,   3.71,   3.56,   3.40,   3.24,   3.09,   2.93,   2.77,
            2.63,   2.47,   2.35,   2.21,   2.09,   1.98,   1.87,   1.78,
            1.69,   1.63,   1.57,   1.51,   1.47,   1.44,   1.40,   1.38,
            1.33,   1.30,   1.26,   1.22,   1.15,   1.06,   0.95,   0.83]

def signext_24x(byte, bits, be):
    padding = int(byte.hex(), base=16) & (1<<be and (bits-1) or 7) and b'\xff' or b'\x00'
    if be: return padding * (bits//24) + byte
    else: return byte + padding * (bits//24)

def signext_12(hex_str, be):
    prefix = be and hex_str[0] or hex_str[2]
    padding = int(prefix, base=16) > 7 and 'f' or '0'
    if be: return padding + hex_str
    else: return hex_str[:2] + padding + hex_str[2]

def quant(freqs, thresholds, kwargs):
    dlen = len(freqs[0])
    fs_list = {n: get_range(dlen, kwargs['sample_rate'], n) for n in subband}

    const_factor = np.log2(kwargs['level']+1)*2+1

    for c in range(len(freqs)):
        for i, j in zip(subband[:-1], subband[1:]):
            af = 2**(-qfactors[subband.index(i)])
            band_freqs = freqs[c][fs_list[i]:fs_list[j]]
            band_thresholds = thresholds[c][fs_list[i]:fs_list[j]]

            mask = np.abs(band_freqs) < band_thresholds * const_factor
            band_freqs[mask] = 0

            freqs[c][fs_list[i]:fs_list[j]] = band_freqs / af

    return freqs

def dequant(freqs, kwargs):
    dlen = len(freqs[0])
    fs_list = {n:get_range(dlen, kwargs['sample_rate'], n) for n in subband}

    for c in range(len(freqs)):
        for i, j in zip(subband[:-1], subband[1:]):
            af = 2**(-qfactors[subband.index(i)])
            freqs[c][fs_list[i]:fs_list[j]] *= af
    return freqs

def analogue(data: np.ndarray, bits: int, channels: int, little_endian: bool, kwargs) -> bytes:
    be = not little_endian
    endian = be and '>' or '<'

    # DCT
    freqs = np.array([dct(data[:, i], norm='ortho') for i in range(channels)])
    dlen = len(data)

    if kwargs['level'] > 10:
        for chnl in range(channels):
            res = int(dlen / kwargs['sample_rate'] * 2 * (24000 - (kwargs['level']-10)*2000))
            freqs[chnl][res:] = 0

    thresholds = l1tools.get_thres(freqs*65536, channels, dlen, kwargs)/65536
    freqs = quant(freqs, thresholds, kwargs)
    # Inter-channel prediction
    freqs[1:] -= freqs[0]

    # Overflow check & Increasing bit depth
    while not (2**(bits-1)-1 >= freqs.any() >= -(2**(bits-1))):
        if bits == 128: raise Exception('Overflow with reaching the max bit depth.')
        bits = {16:24, 24:32, 32:48, 48:64, 64:128}.get(bits, 128)

    # Ravelling and packing
    data: np.ndarray = np.column_stack(np.array(freqs).astype(dtypes[bits]).newbyteorder(endian)).ravel(order='C').tobytes()

    # Cutting off bits
    if bits in [128, 64, 32, 16]:
        pass
    elif bits in [48, 24]:
        data = b''.join([be and data[i+(bits//24):i+(bits//6)] or data[i:i+(bits//8)] for i in range(0, len(data), bits//6)])
    elif bits == 12:
        data = data.hex()
        data = bytes.fromhex(''.join([be and data[i+1:i+4] or data[i:i+4][:2] + data[i:i+4][3:] for i in range(0, len(data), 4)]))
    else: raise Exception('Illegal bits value.')

    # Deflating
    data = zlib.compress(data, level=9)

    return data, bits, channels

def digital(data: bytes, fb: int, channels: int, little_endian: bool, *, kwargs) -> np.ndarray:
    be = not little_endian
    endian = be and '>' or '<'
    bits = {0b110:128,0b101:64,0b100:48,0b011:32,0b010:24,0b001:16,0b000:12}[fb]

    # Inflating
    data = zlib.decompress(data)

    # Padding bits
    if bits % 3 != 0: pass
    elif bits in [24, 48]:
        data = b''.join([signext_24x(data[i:i+(bits//8)], bits, be) for i in range(0, len(data), bits//8)])
    elif bits == 12:
        data = data.hex()
        data = ''.join([signext_12(data[i:i+3], be) for i in range(0, len(data), 3)])
        data = bytes.fromhex(data)
    else:
        raise Exception('Illegal bits value.')

    # Unpacking and unravelling
    data = np.frombuffer(data, dtype=endian+dtypes[bits]).astype(float)
    freqs = [data[i::channels] for i in range(channels)]

    # Removing potential Infinities and Non-numbers
    freqs = np.where(np.isnan(freqs) | np.isinf(freqs), 0, freqs)
    # Inter-channel reconstruction
    freqs[1:] += freqs[0]
    freqs = dequant(freqs, kwargs)

    # Inverse DCT and stacking
    return np.column_stack([idct(chnl, norm='ortho') for chnl in freqs])
