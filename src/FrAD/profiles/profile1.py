from scipy.fft import dct, idct
import numpy as np
from .tools import profile1 as p1tools
import zlib

dtypes = {128:'i16',64:'i8',48:'i8',32:'i4',24:'i4',16:'i2',12:'i2'}
get_range = lambda fs, sr, x: x is not np.inf and int(fs*x*2/sr+0.5) or 2**32
subbands = [0,     200,   400,   600,   800,   1000,  1200,  1400,
            1600,  2000,  2400,  2800,  3200,  4000,  4800,  5600,
            6800,  8000,  9600,  12000, 15600, 20000, 25000, 31000,
            38400, 48000, np.inf]
qfactors = [4.06,  4.68,  5.31,  5.93,  6.49,  7.00,  7.14,  7.38,
            7.66,  7.82,  7.76,  7.56,  7.63,  7.37,  7.12,  7.07,
            6.80,  6.32,  6.09,  5.46,  5.24,  4.63,  3.86,  3.43,
            3.25,  2.94]

def signext_24x(byte: bytes, bits, be):
    byte = byte.hex()
    prefix = be and byte[0] or byte[-2]
    padding = int(prefix, base=16) > 7 and 'f' or '0'
    if be: return bytes.fromhex(padding * (bits//12) + byte)
    else: return bytes.fromhex(byte + padding * (bits//12))

def signext_12(hex_str, be):
    prefix = be and hex_str[0] or hex_str[2]
    padding = int(prefix, base=16) > 7 and 'f' or '0'
    if be: return padding + hex_str
    else: return hex_str[:2] + padding + hex_str[2]

def quant(freqs, thresholds, kwargs):
    dlen = len(freqs[0])
    fs_list = {n: get_range(dlen, kwargs['sample_rate'], n) for n in subbands}

    const_factor = np.log2(kwargs['level']+1)*2+1

    for c in range(len(freqs)):
        for i, j in zip(subbands[:-1], subbands[1:]):
            af = 2**(-qfactors[subbands.index(i)])
            band_freqs = freqs[c][fs_list[i]:fs_list[j]]
            band_thresholds = thresholds[c][fs_list[i]:fs_list[j]]

            mask = np.abs(band_freqs) < band_thresholds * const_factor
            band_freqs[mask] = 0

            freqs[c][fs_list[i]:fs_list[j]] = band_freqs / af

    return freqs

def dequant(freqs, kwargs):
    dlen = len(freqs[0])
    fs_list = {n:get_range(dlen, kwargs['sample_rate'], n) for n in subbands}

    for c in range(len(freqs)):
        for i, j in zip(subbands[:-1], subbands[1:]):
            af = 2**(-qfactors[subbands.index(i)])
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

    thresholds = p1tools.get_thres(freqs*65536, channels, dlen, kwargs)/65536
    freqs = quant(freqs, thresholds, kwargs)
    # Inter-channel prediction
    if channels == 1: pass
    elif channels == 2:
        freqs = np.array([(freqs[0] + freqs[1]) / 2, (freqs[0] - freqs[1]) / 2])
    else:
        mid = np.mean(freqs, axis=0)
        freqs = np.vstack([mid, freqs-mid])

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
        data = data.hex()
        data = bytes.fromhex(''.join([be and data[i+(bits//12):i+(bits//6*2)] or data[i:i+bits//4] for i in range(0, len(data), bits//6*2)]))
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
    if channels > 2: channels += 1
    freqs = [data[i::channels] for i in range(channels)]

    # Removing potential Infinities and Non-numbers
    freqs = np.where(np.isnan(freqs) | np.isinf(freqs), 0, freqs)
    # Inter-channel reconstruction
    
    if channels == 1: pass
    elif channels == 2:
        freqs = np.array([freqs[0] + freqs[1], freqs[0] - freqs[1]])
    else: freqs = freqs[1:] + freqs[0]
    freqs = dequant(freqs, kwargs)

    # Inverse DCT and stacking
    return np.column_stack([idct(chnl, norm='ortho') for chnl in freqs])
