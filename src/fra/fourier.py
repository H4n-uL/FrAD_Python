from scipy.fft import dct, idct
import numpy as np
from .tools.psycho import loss
import zlib

class fourier:
    dtypes = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2',12:'f2'}

    def analogue(data: np.ndarray, bits: int, channels: int, little_endian: bool, *, lossy: bool = None, **kwargs):
        be = not little_endian
        endian = be and '>' or '<'
        freqs = [dct(data[:, i]) for i in range(channels)]
        dlen = len(data)

        if lossy:
            freqs = np.sign(freqs) * np.abs(freqs)**(3/4)
            freqs = loss.filter(freqs, channels, dlen, kwargs)

        while any(np.max(np.abs(c)) > np.finfo(fourier.dtypes[bits]).max for c in freqs):
            if bits == 128: raise Exception('Overflow with reaching the max bit depth.')
            bits = {16:24, 24:32, 32:48, 48:64, 64:128}.get(bits, 128)

        data: bytes = np.column_stack([chnl.astype(fourier.dtypes[bits]).newbyteorder(endian) for chnl in freqs]).ravel(order='C').tobytes()
        if bits in [128, 64, 32, 16]:
            pass
        elif bits in [48, 24]:
            data = b''.join([be and data[i:i+(bits//8)] or data[i+(bits//24):i+(bits//6)] for i in range(0, len(data), bits//6)])
        elif bits == 12:
            data = data.hex()
            data = bytes.fromhex(''.join([be and data[i:i+3] or data[i:i+4][0] + data[i:i+4][2:] for i in range(0, len(data), 4)]))
        else: raise Exception('Illegal bits value.')

        if lossy: data = zlib.compress(data, level=9)

        return data, bits

    def digital(data: bytes, fb: int, channels: int, little_endian: bool, *, lossy: bool = None):
        if lossy: data = zlib.decompress(data)
        be = not little_endian
        endian = be and '>' or '<'
        bits = {0b110:128,0b101:64,0b100:48,0b011:32,0b010:24,0b001:16,0b000:12}[fb]
        if bits % 3 != 0: pass
        elif bits in [24, 48]:
            data = b''.join([be and data[i:i+(bits//8)]+(b'\x00'*(bits//24)) or (b'\x00'*(bits//24))+data[i:i+(bits//8)] for i in range(0, len(data), bits//8)])
        elif bits == 12:
            data = data.hex()
            if endian == '<': data = ''.join([data[i:i+3][0] + '0' + data[i:i+3][1:] for i in range(0, len(data), 3)])
            else: data = ''.join([data[i:i+3] + '0' for i in range(0, len(data), 3)])
            data = bytes.fromhex(data)
        else:
            raise Exception('Illegal bits value.')
        data_numpy = np.frombuffer(data, dtype=endian+fourier.dtypes[bits]).astype(float)

        freqs = [data_numpy[i::channels] for i in range(channels)]
        freqs = np.where(np.isnan(freqs) | np.isinf(freqs), 0, freqs)
        if lossy: freqs = np.sign(freqs) * np.abs(freqs)**(4/3)

        return np.column_stack([idct(chnl) for chnl in freqs])

    get_range = lambda fs, sr, x: x is not np.inf and int(fs*x*2/sr+0.5) or 2**32
