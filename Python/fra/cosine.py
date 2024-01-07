import math
from ml_dtypes import bfloat16
from mdctn import mdct, imdct
import numpy as np

class cosine:
    def analogue(data, bits: int, channels: int):
        odd = len(data[:, 0]) % 2 != 0
        fft_data = [mdct(np.concatenate((data[:, i], [0])) if odd else data[:, i], N=math.ceil(len(data)/2)*2) for i in range(channels)]

        # if bits == 512: freq = [d.astype(np.float512) for d in fft_data]
        # elif bits == 256: freq = [d.astype(np.float256) for d in fft_data]
        # elif bits == 128: freq = [d.astype(np.float128) for d in fft_data]
        if bits == 64: freq = [d.astype(np.float64) for d in fft_data]
        elif bits == 32: freq = [d.astype(np.float32) for d in fft_data]
        elif bits == 16: freq = [d.astype(bfloat16) for d in fft_data]
        else: raise Exception('Illegal bits value.')

        data = np.column_stack(freq).ravel(order='C').tobytes()
        return data, odd

    def digital(data, fb: int, bits: int, channels: int, unpad: bool):
        # if fb == 0b110: data_numpy = np.frombuffer(data, dtype=np.float512)
        # elif fb == 0b101: data_numpy = np.frombuffer(data, dtype=np.float256)
        # elif fb == 0b100: data_numpy = np.frombuffer(data, dtype=np.float128)
        if fb == 0b011: data_numpy = np.frombuffer(data, dtype=np.float64)
        elif fb == 0b010: data_numpy = np.frombuffer(data, dtype=np.float32)
        elif fb == 0b001: data_numpy = np.frombuffer(data, dtype=bfloat16)
        else:
            raise Exception('Illegal bits value.')

        freq = [data_numpy[i::channels] for i in range(channels)]
        if unpad:
            wave_data = [np.int32(np.clip(imdct(d, N=len(d))[:-1], -2**31, 2**31-1)) for d in freq]
        else:
            wave_data = [np.int32(np.clip(imdct(d, N=len(d)), -2**31, 2**31-1)) for d in freq]

        if bits == 32: pass
        elif bits == 16: wave_data = [np.int16(wave / 2**16) for wave in wave_data]
        elif bits == 8: wave_data = [np.uint8(wave / 2**24 + 2**7) for wave in wave_data]
        else:
            raise ValueError(f"Illegal value {bits} for bits: only 8, 16, and 32 bits are available for decoding.")

        return np.column_stack(wave_data)
