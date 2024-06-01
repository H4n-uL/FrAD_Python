from scipy.fft import dct, idct
import numpy as np
from .tools import p1tools
from ..tools.headb import headb
import zlib

depths = [8, 12, 16, 24, 32, 48, 64]
dtypes = {64:'i8',48:'i8',32:'i4',24:'i4',16:'i2',12:'i2',8:'i1'}
get_range = lambda fs, sr, x: x is not np.inf and int(fs*x*2/sr+0.5) or 2**32

def signext_24x(byte: bytes, bits, be):
    return (int((be and byte.hex()[0] or byte.hex()[-1]), base=16) > 7 and b'\xff' or b'\x00') * (bits//24) + byte

def signext_12(hex_str):
    if len(hex_str)!=3: return ''
    return (int(hex_str[0], base=16) > 7 and 'f' or '0') + hex_str

def analogue(pcm: np.ndarray, bits: int, channels: int, little_endian: bool, kwargs) -> tuple[bytes, int, int, int]:
    be = not little_endian
    endian = be and '>' or '<'

    # DCT
    dlen = len(pcm)
    pcm = np.pad(pcm, ((0, headb.decode_css_prf1(headb.encode_css_prf1(channels, 48000, dlen))[2]-dlen), (0, 0)), mode='constant')
    dlen = len(pcm)
    freqs = np.array([dct(pcm[:, i]*(2**(bits-1))) for i in range(channels)]) / dlen

    freqs, pns = p1tools.quant(freqs, channels, dlen, kwargs)

    # Overflow check & Increasing bit depth
    while not (2**(bits-1)-1 >= freqs.any() >= -(2**(bits-1))):
        if bits == 64: raise Exception('Overflow with reaching the max bit depth.')
        bits = {8:12, 12:16, 16:24, 24:32, 32:48, 48:64}.get(bits, 64)

    # Ravelling and packing
    if bits%8!=0: endian = '>'
    frad: bytes = freqs.T.ravel().astype(endian+dtypes[bits]).tobytes()

    # Cutting off bits
    if bits in [64, 32, 16, 8]:
        pass
    elif bits in [48, 24]:
        hexa = frad.hex()
        frad = bytes.fromhex(''.join([be and hexa[i+(bits//12):i+(bits//6*2)] or hexa[i:i+bits//4] for i in range(0, len(hexa), bits//6*2)]))
    elif bits == 12:
        hexa = frad.hex()
        hexa = ''.join([hexa[i+1:i+4] for i in range(0, len(hexa), 4)])
        if len(hexa)%2!=0: hexa+='0'
        frad = bytes.fromhex(hexa)
    else: raise Exception('Illegal bits value.')

    frad = (pns.T/(2**(bits-1))).astype(endian+'e').tobytes() + frad

    # Deflating
    frad = zlib.compress(frad, level=9)

    return frad, bits, channels, depths.index(bits)

def digital(frad: bytes, fb: int, channels: int, little_endian: bool, kwargs) -> np.ndarray:
    be = not little_endian
    endian = be and '>' or '<'
    bits = depths[fb]


    # Inflating
    frad = zlib.decompress(frad)
    thresbytes = frad[:p1tools.subbands*channels*2]
    thres = np.frombuffer(thresbytes, dtype=endian+'e').reshape((-1, channels)).T * (2**(bits-1))
    frad = frad.removeprefix(thresbytes)

    # Padding bits
    if bits % 3 != 0: pass
    elif bits in [24, 48]:
        frad = b''.join([signext_24x(frad[i:i+(bits//8)], bits, be) for i in range(0, len(frad), bits//8)])
    elif bits == 12:
        hexa = frad.hex()
        frad = bytes.fromhex(''.join([signext_12(hexa[i:i+3]) for i in range(0, len(hexa), 3)]))
    else: raise Exception('Illegal bits value.')

    # Unpacking and unravelling
    if channels > 2: channels += 1
    if bits%8!=0: endian = '>'
    freqs: np.ndarray = np.frombuffer(frad, dtype=endian+dtypes[bits]).astype(float).reshape(-1, channels).T

    # Removing potential Infinities and Non-numbers
    freqs = np.where(np.isnan(freqs) | np.isinf(freqs), 0, freqs)
    freqs = p1tools.dequant(freqs, channels, thres, kwargs)

    # Inverse DCT and stacking
    return np.ascontiguousarray(np.array([idct(chnl*len(chnl)) for chnl in freqs]).T)/(2**(bits-1))
