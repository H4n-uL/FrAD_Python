from mdctn import mdct, imdct
import numpy as np
from .tools.lossy_psycho import psycho
nfilts=64

class fourier:
    def analogue(data: np.ndarray, bits: int, channels: int, little_endian: bool, *, lossy: bool, sample_rate: int, level: int):
        be = not little_endian
        endian = be and '>' or '<'
        dt = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2',12:'f2'}[bits]
        data = np.pad(data, ((0, -len(data[:, 0])%2), (0, 0)), mode='constant')
        fft_data = [mdct(data[:, i], N=len(data)*2) for i in range(channels)]

        # MARK: Temporary Psychoacoustic Filtering
        if lossy:
            frame_size, alpha = len(data), (800 - (1.2**level))*0.001
            W = psycho.mapping2barkmat(sample_rate,nfilts,frame_size*2)
            W_inv = psycho.mappingfrombarkmat(W,frame_size*2)
            sprfuncBarkdB = psycho.f_SP_dB(sample_rate/2,nfilts)
            sprfuncmat = psycho.sprfuncmat(sprfuncBarkdB, alpha, nfilts)
            for c in range(channels):
                fft_data[c] = np.around(fft_data[c] / (frame_size / 16384)) * (frame_size / 16384)
                mXbark = psycho.mapping2bark(np.abs(fft_data[c]),W,frame_size*2)
                mTbark = psycho.maskingThresholdBark(mXbark,sprfuncmat,alpha,sample_rate,nfilts) * np.log2(level+1)/2
                thres =  psycho.mappingfrombark(mTbark,W_inv,frame_size*2)[:-1] * np.linspace(0.25-(level/10), 0.25+((sample_rate/48000)*(level/5)), len(fft_data[c]))
                fft_data[c][np.abs(fft_data[c]) < thres] = 0
        # ENDMARK

        while any(np.max(np.abs(c)) > np.finfo(dt).max for c in fft_data):
            if bits == 128: raise Exception('Overflow with reaching the max bit depth.')
            bits = {16:24, 24:32, 32:48, 48:64, 64:128}.get(bits, 128) 
            dt = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2',12:'f2'}[bits]

        data: bytes = np.column_stack([d.astype(dt).newbyteorder(endian) for d in fft_data]).ravel(order='C').tobytes()
        if bits in [64, 32, 16]:
            pass
        elif bits in [48, 24]:
            data = b''.join([be and data[i:i+(bits//8)] or data[i+(bits//24):i+(bits//6)] for i in range(0, len(data), bits//6)])
        elif bits == 12:
            data = data.hex()
            data = bytes.fromhex(''.join([be and data[i:i+3] or data[i:i+4][0] + data[i:i+4][2:] for i in range(0, len(data), 4)]))
        else: raise Exception('Illegal bits value.')

        return data, bits

    def digital(data: bytes, fb: int, channels: int, little_endian: bool):
        be = not little_endian
        endian = be and '>' or '<'
        dt = {0b101:'d',0b100:'d',0b011:'f',0b010:'f',0b001:'e',0b000:'e'}[fb]
        if fb in [0b101,0b011,0b001]:
            pass
        elif fb in [0b100,0b010]:
            if fb == 0b100: data = b''.join([be and data[i:i+6]+b'\x00\x00' or b'\x00\x00'+data[i:i+6] for i in range(0, len(data), 6)])
            elif fb == 0b010: data = b''.join([be and data[i:i+3]+b'\x00' or b'\x00'+data[i:i+3] for i in range(0, len(data), 3)])
        elif fb == 0b000:
            data = data.hex()
            if endian == '<': data = ''.join([data[i:i+3][0] + '0' + data[i:i+3][1:] for i in range(0, len(data), 3)])
            else: data = ''.join([data[i:i+3] + '0' for i in range(0, len(data), 3)])
            data = bytes.fromhex(data)
        else:
            raise Exception('Illegal bits value.')
        data_numpy = np.frombuffer(data, dtype=endian+dt).astype(float)

        freq = [data_numpy[i::channels] for i in range(channels)]
        freq = np.where(np.isnan(freq) | np.isinf(freq), 0, freq)

        return np.column_stack([imdct(d, N=len(d)*2) for d in freq])
