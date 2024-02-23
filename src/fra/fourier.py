from mdctn import mdct, imdct
import numpy as np
from .tools.lossy_psycho import psycho
nfilts=64
alpha=0.8

class fourier:
    def analogue(data: np.ndarray, bits: int, channels: int, big_endian: bool, lossy: bool, sample_rate: int, level: int):
        if lossy:
            block_size = len(data)
            W = psycho.mapping2barkmat(sample_rate,nfilts,block_size*2)
            W_inv = psycho.mappingfrombarkmat(W,block_size*2)
            sprfuncBarkdB = psycho.f_SP_dB(sample_rate/2,nfilts)
            sprfuncmat = psycho.sprfuncmat(sprfuncBarkdB, alpha, nfilts)

        endian = big_endian and '>' or '<'
        dt = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2'}[bits]
        data = np.pad(data, ((0, -len(data[:, 0])%2), (0, 0)), mode='constant')
        fft_data = [mdct(data[:, i], N=len(data)*2) for i in range(channels)]

        if lossy:
            v = len(fft_data) // 16
            for c in range(channels):
                mXbark = psycho.mapping2bark(np.abs(fft_data[c]),W,block_size*2)
                mTbark = psycho.maskingThresholdBark(mXbark,sprfuncmat,alpha,sample_rate,nfilts)
                thres =  (psycho.mappingfrombark(mTbark,W_inv,block_size*2) / 4) ** (1.01*(level/3)+1)
                fft_data[c][v:][np.abs(fft_data[c][v:]) < thres[v:-1]] = 0

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
