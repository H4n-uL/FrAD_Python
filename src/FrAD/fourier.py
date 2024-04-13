from scipy.fft import dct, idct
import numpy as np
from .layers import layer1

class fourier:
    dtypes = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2',12:'f2'}

    def analogue(data: np.ndarray, bits: int, channels: int, little_endian: bool, *, layer: int = 0, **kwargs) -> bytes:
        if layer == 1: return layer1.analogue(data, bits, channels, little_endian, kwargs)

        be = not little_endian
        endian = be and '>' or '<'

        # DCT
        freqs = np.array([dct(data[:, i]) for i in range(channels)])

        # Overflow check & Increasing bit depth
        while np.max(np.abs(freqs)) > np.finfo(fourier.dtypes[bits]).max:
            if bits == 128: raise Exception('Overflow with reaching the max bit depth.')
            bits = {12:16, 16:24, 24:32, 32:48, 48:64, 64:128}.get(bits, 128)

        # Ravelling and packing
        data: np.ndarray = np.column_stack(freqs.astype(fourier.dtypes[bits]).newbyteorder(endian)).ravel(order='C').tobytes()

        # Cutting off bits
        if bits in [128, 64, 32, 16]:
            pass
        elif bits in [48, 24]:
            data = b''.join([be and data[i:i+(bits//8)] or data[i+(bits//24):i+(bits//6)] for i in range(0, len(data), bits//6)])
        elif bits == 12:
            data = data.hex()
            data = bytes.fromhex(''.join([be and data[i:i+3] or data[i:i+4][0] + data[i:i+4][2:] for i in range(0, len(data), 4)]))
        else: raise Exception('Illegal bits value.')

        return data, bits, channels

    def digital(data: bytes, fb: int, channels: int, little_endian: bool, *, layer: int = 0, **kwargs) -> np.ndarray:
        if layer == 1: return layer1.digital(data, fb, channels, little_endian, kwargs=kwargs)

        be = not little_endian
        endian = be and '>' or '<'
        bits = {0b110:128,0b101:64,0b100:48,0b011:32,0b010:24,0b001:16,0b000:12}[fb]

        # Padding bits
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

        # Unpacking and unravelling
        data = np.frombuffer(data, dtype=endian+fourier.dtypes[bits]).astype(float)
        freqs = [data[i::channels] for i in range(channels)]

        # Removing potential Infinities and Non-numbers
        freqs = np.where(np.isnan(freqs) | np.isinf(freqs), 0, freqs)

        # Inverse DCT and stacking
        return np.column_stack([idct(chnl) for chnl in freqs])
