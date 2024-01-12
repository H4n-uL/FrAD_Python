from ml_dtypes import bfloat16
import numpy as np
from scipy.fft import fft, ifft
import struct

class fourier:
    def analogue(data, bits: int, channels: int):
        fft_data = [fft(data[:, i]) for i in range(channels)]

        # if bits == 128: freq = [np.column_stack((np.abs(d).astype(np.float128), np.angle(d).astype(np.float128))) for d in fft_data]
        if bits == 64: data = np.column_stack([np.column_stack((np.abs(d).astype(np.float64), np.angle(d).astype(np.float64))) for d in fft_data]).ravel(order='C').tobytes()
        elif bits == 48:
            freq = [np.column_stack((np.abs(d).astype(np.float64), np.angle(d).astype(np.float64))) for d in fft_data]
            data = b''.join([struct.pack('<d', d)[2:] for d in np.column_stack(freq).ravel(order='C')])
        elif bits == 32: data = np.column_stack([np.column_stack((np.abs(d).astype(np.float32), np.angle(d).astype(np.float32))) for d in fft_data]).ravel(order='C').tobytes()
        elif bits == 24:
            freq = [np.column_stack((np.abs(d).astype(np.float32), np.angle(d).astype(np.float32))) for d in fft_data]
            data = b''.join([struct.pack('<f', d)[1:] for d in np.column_stack(freq).ravel(order='C')])
        elif bits == 16: data = np.column_stack([np.column_stack((np.abs(d).astype(bfloat16), np.angle(d).astype(bfloat16))) for d in fft_data]).ravel(order='C').tobytes()
        else: raise Exception('Illegal bits value.')

        return data

    def digital(data, fb: int, bits: int, channels: int):
        # if fb == 0b110: data_numpy = np.frombuffer(data, dtype=np.float128)
        if fb == 0b101: data_numpy = np.frombuffer(data, dtype=np.float64)
        elif fb == 0b100:
            data = b''.join([b'\x00\x00'+data[i:i+6] for i in range(0, len(data), 6)])
            data_numpy = np.frombuffer(data, dtype='<d')
        elif fb == 0b011: data_numpy = np.frombuffer(data, dtype=np.float32)
        elif fb == 0b010:
            data = b''.join([b'\x00'+data[i:i+3] for i in range(0, len(data), 3)])
            data_numpy = np.frombuffer(data, dtype='<f')
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
