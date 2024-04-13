from scipy.fft import dct, idct
import numpy as np
from ..tools.psycho import loss
import zlib

dtypes = {128:'i16',64:'i8',48:'i8',32:'i4',24:'i4',16:'i2',12:'i2'}

fl = [0, 100, 500, 2000, 5000, 10000, 20000, 100000, 500000, np.inf]
rfs = [4, 5, 7, 8, 6, 5, 4, 2, 0]

def signext_24x(byte, bits, be):
    padding = int(byte.hex(), base=16) & (1<<be and (bits-1) or 7) and b'\xff' or b'\x00'
    if be: return padding * (bits//24) + byte
    else: return byte + padding * (bits//24)

def signext_12(hex_str, be):
    prefix = be and hex_str[0] or hex_str[2]
    padding = int(prefix, base=16) > 7 and 'f' or '0'
    if be: return padding + hex_str
    else: return hex_str[:2] + padding + hex_str[2]

def rounding(freqs, kwargs):
    dlen = len(freqs[0])
    fs_list = {n:loss.get_range(dlen, kwargs['sample_rate'], n) for n in fl}

    for c in range(len(freqs)):
        for i, j in zip(fl[:-1], fl[1:]):
            af = 2**(-rfs[fl.index(i)])
            freqs[c][fs_list[i]:fs_list[j]] /= af
    return freqs

def unrounding(freqs, kwargs):
    dlen = len(freqs[0])
    fs_list = {n:loss.get_range(dlen, kwargs['sample_rate'], n) for n in fl}

    for c in range(len(freqs)):
        for i, j in zip(fl[:-1], fl[1:]):
            af = 2**(-rfs[fl.index(i)])
            freqs[c][fs_list[i]:fs_list[j]] *= af
    return freqs

def analogue(data: np.ndarray, bits: int, channels: int, little_endian: bool, kwargs) -> bytes:
    be = not little_endian
    endian = be and '>' or '<' # DCT
    freqs = np.array([dct(data[:, i], norm='ortho') for i in range(channels)])
    dlen = len(data)

    freqs = np.sign(freqs) * np.abs(freqs)**(3/4)
    freqs = loss.filter(freqs*65536, channels, dlen, kwargs)/65536
    freqs = rounding(freqs, kwargs)
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
    freqs = unrounding(freqs, kwargs)
    # Inter-channel reconstruction
    freqs[1:] += freqs[0]
    freqs = np.sign(freqs) * np.abs(freqs)**(4/3)

    # Inverse DCT and stacking
    return np.column_stack([idct(chnl, norm='ortho') for chnl in freqs])
