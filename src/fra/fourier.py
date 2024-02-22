from mdctn import mdct, imdct
import numpy as np

class fourier:
    def analogue(data: np.ndarray, bits: int, channels: int, big_endian: bool):
        endian = big_endian and '>' or '<'
        dt = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2'}[bits]
        data = np.pad(data, ((0, -len(data[:, 0])%2), (0, 0)), mode='constant')
        fft_data = [mdct(data[:, i], N=len(data)*2) for i in range(channels)]

        while any(np.max(np.abs(c)) > np.finfo(dt).max for c in fft_data):
            if bits == 128: raise Exception('Overflow with reaching the max bit depth.')
            bits = {16:24, 24:32, 32:48, 48:64, 64:128}.get(bits, 128) 
            dt = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2'}[bits]

        data = np.column_stack([d.astype(dt).newbyteorder(endian) for d in fft_data]).ravel(order='C').tobytes()
        if bits in [64, 32, 16]:
            pass
        elif bits in [48, 24]:
            data = b''.join([big_endian and data[i:i+(bits//8)] or data[i+(bits//24):i+(bits//6)] for i in range(0, len(data), bits//6)])
        else: raise Exception('Illegal bits value.')

        return data, bits

    def analogue_lc(data: np.ndarray, bits: int, channels: int, big_endian: bool, sample_rate: int, level: int):
        block_size = len(data)
        f = np.linspace(0, sample_rate/2, block_size)

        LTQ=-6.5*np.exp(-0.6*(f/1000.-3.3)**2.)+1e-3*((f/1000.)**3)
        thres = 10 ** (LTQ / 20) / 40 * np.log2(block_size) * 1.2**level

        endian = big_endian and '>' or '<'
        dt = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2'}[bits]
        data = np.pad(data, ((0, -len(data[:, 0])%4), (0, 0)), mode='constant')

        fft_data = [mdct(data[:, i], N=len(data)*2) for i in range(channels)]
        for i in range(channels):
            fft_data[i][np.abs(fft_data[i]) < thres] = 0

        while any(np.max(np.abs(c)) > np.finfo(dt).max for c in fft_data):
            if bits == 128: raise Exception('Overflow with reaching the max bit depth.')
            bits = {16:24, 24:32, 32:48, 48:64, 64:128}.get(bits, 128) 
            dt = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2'}[bits]

        data = np.column_stack([d.astype(dt).newbyteorder(endian) for d in fft_data]).ravel(order='C').tobytes()
        if bits in [64, 32, 16]:
            pass
        elif bits in [48, 24]:
            data = b''.join([big_endian and data[i:i+(bits//8)] or data[i+(bits//24):i+(bits//6)] for i in range(0, len(data), bits//6)])
        else: raise Exception('Illegal bits value.')

        return data, bits

    def digital(data: bytes, fb: int, channels: int, big_endian: bool):
        endian = big_endian and '>' or '<'
        dt = {0b101:'d',0b100:'d',0b011:'f',0b010:'f',0b001:'e'}[fb]
        if fb in [0b101,0b011,0b001]:
            pass
        elif fb in [0b100,0b010]:
            if fb == 0b100: data = b''.join([big_endian and data[i:i+6]+b'\x00\x00' or b'\x00\x00'+data[i:i+6] for i in range(0, len(data), 6)])
            elif fb == 0b010: data = b''.join([big_endian and data[i:i+3]+b'\x00' or b'\x00'+data[i:i+3] for i in range(0, len(data), 3)])
        else:
            raise Exception('Illegal bits value.')
        data_numpy = np.frombuffer(data, dtype=endian+dt).astype(float)

        freq = [data_numpy[i::channels] for i in range(channels)]
        freq = np.where(np.isnan(freq) | np.isinf(freq), 0, freq)

        return np.column_stack([imdct(d, N=len(d)*2) for d in freq])
