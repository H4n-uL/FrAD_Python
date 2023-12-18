from ml_dtypes import bfloat16
import numpy as np
from scipy.fft import fft, ifft

class fourier:
    def analogue(data, bits: int, channels: int, osr: int, nsr: int = None):
        fft_data = [fft(data[:, i]) for i in range(channels)]

        # if bits == 512: freq = [np.column_stack((np.abs(d).astype(np.float512), np.angle(d).astype(np.float512))) for d in fft_data]
        # elif bits == 256: freq = [np.column_stack((np.abs(d).astype(np.float256), np.angle(d).astype(np.float256))) for d in fft_data]
        # elif bits == 128: freq = [np.column_stack((np.abs(d).astype(np.float128), np.angle(d).astype(np.float128))) for d in fft_data]
        if bits == 64: freq = [np.column_stack((np.abs(d).astype(np.float64), np.angle(d).astype(np.float64))) for d in fft_data]
        elif bits == 32: freq = [np.column_stack((np.abs(d).astype(np.float32), np.angle(d).astype(np.float32))) for d in fft_data]
        elif bits == 16: freq = [np.column_stack((np.abs(d).astype(bfloat16), np.angle(d).astype(bfloat16))) for d in fft_data]
        else: raise Exception('Illegal bits value.')

        data = np.column_stack(freq).ravel(order='C').tobytes()
        return data

    def digital(data, fb: int, bits: int, channels: int):
        # if fb == 0b110: data_numpy = np.frombuffer(data, dtype=np.float512)
        # elif fb == 0b101: data_numpy = np.frombuffer(data, dtype=np.float256)
        # elif fb == 0b100: data_numpy = np.frombuffer(data, dtype=np.float128)
        if fb == 0b011: data_numpy = np.frombuffer(data, dtype=np.float64)
        elif fb == 0b010: data_numpy = np.frombuffer(data, dtype=np.float32)
        elif fb == 0b001: data_numpy = np.frombuffer(data, dtype=bfloat16)
        else:
            raise Exception('Illegal bits value.')

        data_numpy = data_numpy.reshape(-1, channels*2)
        freq = np.split(data_numpy, channels, axis=1)
        wave_data = [np.int32(np.clip(np.real(ifft(d[:, 0] * np.exp(1j * d[:, 1]))), -2**31, 2**31-1)) for d in freq]

        if bits == 32: pass
        elif bits == 16: wave_data = [np.int16(wave / 2**16) for wave in wave_data]
        elif bits == 8: wave_data = [np.uint8(wave / 2**24 + 2**7) for wave in wave_data]
        else:
            raise ValueError(f"Illegal value {bits} for bits: only 8, 16, and 32 bits are available for decoding.")

        return np.column_stack(wave_data)
