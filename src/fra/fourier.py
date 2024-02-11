import struct
from mdctn import mdct, imdct
import numpy as np

class fourier:
    def analogue(data: np.ndarray, bits: int, channels: int, big_endian: bool):
        endian = big_endian and '>' or '<'
        pad_length = (4 - len(data[:, 0])) % 4
        data = np.pad(data, ((0, pad_length), (0, 0)), mode='constant')
        fft_data = [mdct(data[:, i], N=len(data)) for i in range(channels)]

        # if bits == 128: freq = [d.astype(np.float128) for d in fft_data]
        if bits == 64: data = np.column_stack([d.astype(np.float64).newbyteorder(endian) for d in fft_data]).ravel(order='C').tobytes()
        elif bits == 48:
            freq = [d.astype(np.float64) for d in fft_data]
            if big_endian:
                data = b''.join([struct.pack('>d', d)[:-2] for d in np.column_stack(freq).ravel(order='C')])
            else:
                data = b''.join([struct.pack('<d', d)[2:] for d in np.column_stack(freq).ravel(order='C')])
        elif bits == 32: data = np.column_stack([d.astype(np.float32).newbyteorder(endian) for d in fft_data]).ravel(order='C').tobytes()
        elif bits == 24:
            freq = [d.astype(np.float32) for d in fft_data]
            if big_endian:
                data = b''.join([struct.pack('>f', d)[:-1] for d in np.column_stack(freq).ravel(order='C')])
            else:
                data = b''.join([struct.pack('<f', d)[1:] for d in np.column_stack(freq).ravel(order='C')])
        elif bits == 16: data = np.column_stack([d.astype(np.float16).newbyteorder(endian) for d in fft_data]).ravel(order='C').tobytes()
        else: raise Exception('Illegal bits value.')

        return data

    def digital(data: np.ndarray, fb: int, channels: int, big_endian: bool):
        if big_endian:
            # if fb == 0b110: data_numpy = np.frombuffer(data, dtype=np.float128).byteswap()
            if fb == 0b101: data_numpy = np.frombuffer(data, dtype=np.float64).byteswap()
            elif fb == 0b100:
                data = b''.join([data[i:i+6]+b'\x00\x00' for i in range(0, len(data), 6)])
                data_numpy = np.frombuffer(data, dtype='>d')
            elif fb == 0b011: data_numpy = np.frombuffer(data, dtype=np.float32).byteswap()
            elif fb == 0b010:
                data = b''.join([data[i:i+3]+b'\x00' for i in range(0, len(data), 3)])
                data_numpy = np.frombuffer(data, dtype='>f')
            elif fb == 0b001: data_numpy = np.frombuffer(data, dtype=np.float16).byteswap()
            else:
                raise Exception('Illegal bits value.')
        else:
            # if fb == 0b110: data_numpy = np.frombuffer(data, dtype=np.float128)
            if fb == 0b101: data_numpy = np.frombuffer(data, dtype=np.float64)
            elif fb == 0b100:
                data = b''.join([b'\x00\x00'+data[i:i+6] for i in range(0, len(data), 6)])
                data_numpy = np.frombuffer(data, dtype='<d')
            elif fb == 0b011: data_numpy = np.frombuffer(data, dtype=np.float32)
            elif fb == 0b010:
                data = b''.join([b'\x00'+data[i:i+3] for i in range(0, len(data), 3)])
                data_numpy = np.frombuffer(data, dtype='<f')
            elif fb == 0b001: data_numpy = np.frombuffer(data, dtype=np.float16)
            else:
                raise Exception('Illegal bits value.')

        freq = [data_numpy[i::channels] for i in range(channels)]

        return np.column_stack([imdct(d, N=len(d)) for d in freq])
