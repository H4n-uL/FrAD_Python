from scipy.fft import dct, idct
import numpy as np
from .tools import p1tools
import zlib

depths = [8, 12, 16, 24, 32, 48, 64]
dtypes = {64:'i8',48:'i8',32:'i4',24:'i4',16:'i2',12:'i2',8:'i1'}
get_range = lambda fs, sr, x: x is not np.inf and int(fs*x*2/sr+0.5) or 2**32

def signext_24x(byte: bytes, bits, be):
    prefix = be and byte.hex()[0] or byte.hex()[-2]
    padding = int(prefix, base=16) > 7 and b'\xff' or b'\x00'
    if be: return padding * (bits//24) + byte
    else: return byte + padding * (bits//24)

def signext_12(hex_str, be):
    prefix = be and hex_str[0] or hex_str[2]
    padding = int(prefix, base=16) > 7 and 'f' or '0'
    if be: return padding + hex_str
    else: return hex_str[:2] + padding + hex_str[2]

def analogue(data: np.ndarray, bits: int, channels: int, little_endian: bool, kwargs) -> bytes:
    be = not little_endian
    endian = be and '>' or '<'

    # DCT
    dlen = len(data)
    freqs = np.array([dct(data[:, i]*(2**(bits-1))) for i in range(channels)]) / dlen

    freqs, pns = p1tools.quant(freqs, channels, dlen, kwargs)
    # Inter-channel prediction
    if channels == 1: pass
    elif channels == 2:
        freqs = np.array([(freqs[0] + freqs[1]) / 2, (freqs[0] - freqs[1]) / 2])
    else:
        mid = np.mean(freqs, axis=0)
        freqs = np.vstack([mid, freqs-mid])

    # Overflow check & Increasing bit depth
    while not (2**(bits-1)-1 >= freqs.any() >= -(2**(bits-1))):
        if bits == 64: raise Exception('Overflow with reaching the max bit depth.')
        bits = {8:12, 12:16, 16:24, 24:32, 32:48, 48:64}.get(bits, 64)

    # Ravelling and packing
    data: bytes = freqs.T.ravel().astype(endian+dtypes[bits]).tobytes()

    # Cutting off bits
    if bits in [64, 32, 16, 8]:
        pass
    elif bits in [48, 24]:
        data = data.hex()
        data = bytes.fromhex(''.join([be and data[i+(bits//12):i+(bits//6*2)] or data[i:i+bits//4] for i in range(0, len(data), bits//6*2)]))
    elif bits == 12:
        data = data.hex()
        data = bytes.fromhex(''.join([be and data[i+1:i+4] or data[i:i+4][:2] + data[i:i+4][3:] for i in range(0, len(data), 4)]))
    else: raise Exception('Illegal bits value.')

    data = (pns/(2**(bits-1))).astype(endian+'e').tobytes() + data

    # Deflating
    data = zlib.compress(data, level=9)

    return data, bits, channels, depths.index(bits)

def digital(data: bytes, fb: int, channels: int, little_endian: bool, *, kwargs) -> np.ndarray:
    be = not little_endian
    endian = be and '>' or '<'
    bits = [8, 12, 16, 24, 32, 48, 64][fb]

    # Inflating
    data = zlib.decompress(data)
    masks = np.frombuffer(data[:p1tools.nfilts*channels*2], dtype=endian+'e').reshape((channels, -1)) * (2**(bits-1))
    data = data[p1tools.nfilts*channels*2:]

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
    if channels > 2: channels += 1
    freqs: np.ndarray = np.frombuffer(data, dtype=endian+dtypes[bits]).astype(float).reshape(-1, channels).T

    # Removing potential Infinities and Non-numbers
    freqs = np.where(np.isnan(freqs) | np.isinf(freqs), 0, freqs)

    # Inter-channel reconstruction
    if channels == 1: pass
    elif channels == 2:
        freqs = np.array([freqs[0] + freqs[1], freqs[0] - freqs[1]])
    else: freqs = freqs[1:] + freqs[0]
    freqs = p1tools.dequant(freqs, channels, masks, kwargs)

    # Inverse DCT and stacking
    return np.column_stack([idct(chnl*len(chnl)) for chnl in freqs])/(2**(bits-1))
