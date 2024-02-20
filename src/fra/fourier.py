from mdctn import mdct, imdct
import numpy as np

class fourier:
    def analogue(data: np.ndarray, bits: int, channels: int, big_endian: bool):
        endian = big_endian and '>' or '<'
        dt = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2'}[bits]
        data = np.pad(data, ((0, -len(data[:, 0])%4), (0, 0)), mode='constant')
        fft_data = [mdct(data[:, i], N=len(data)*2) for i in range(channels)]

        data = np.column_stack([d.astype(dt).newbyteorder(endian) for d in fft_data]).ravel(order='C').tobytes()
        if bits in [64, 32, 16]:
            pass
        elif bits in [48, 24]:
            data = b''.join([big_endian and data[i:i+(bits//8)] or data[i+(bits//24):i+(bits//6)] for i in range(0, len(data), bits//6)])
        else: raise Exception('Illegal bits value.')

        return data

    def analogue_lc(data: np.ndarray, bits: int, channels: int, big_endian: bool, sample_rate: int):
        block_size = len(data)
        f = np.linspace(0, sample_rate/2, block_size)

        LTQ=-6.5*np.exp(-0.6*(f/1000.-3.3)**2.)+1e-3*((f/1000.)**3.4)
        thres = 10 ** (LTQ / 20) / 8

        endian = big_endian and '>' or '<'
        dt = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2'}[bits]
        data = np.pad(data, ((0, -len(data[:, 0])%4), (0, 0)), mode='constant')
        # window = np.hanning(block_size)
        # for i in range(channels):
        #     data[:, i] *= window

        fft_data = [mdct(data[:, i], N=len(data)*2) for i in range(channels)]

        for i in range(channels):
            fft_data[i][np.abs(fft_data[i]) < thres] = 0

        data = np.column_stack([d.astype(dt).newbyteorder(endian) for d in fft_data]).ravel(order='C').tobytes()
        if bits in [64, 32, 16]:
            pass
        elif bits in [48, 24]:
            data = b''.join([big_endian and data[i:i+(bits//8)] or data[i+(bits//24):i+(bits//6)] for i in range(0, len(data), bits//6)])
        else: raise Exception('Illegal bits value.')

        return data

    def digital(data: bytes, fb: int, channels: int, big_endian: bool, prev):
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
        data = np.column_stack([imdct(d, N=len(d)*2) for d in freq])

        if prev is not None:
            for i in range(channels):
                data[0][i] = (prev[i] + data[1][i])/2

        return data, data[-1]
