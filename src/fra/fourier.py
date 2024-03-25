from mdctn import mdct, imdct
import numpy as np
from .tools.lossy_psycho import psycho, PsychoacousticModel
import zlib

class fourier:
    dtypes = {128:'f16',64:'f8',48:'f8',32:'f4',24:'f4',16:'f2',12:'f2'}

    def analogue(data: np.ndarray, bits: int, channels: int, little_endian: bool, *, lossy: bool = None, sample_rate: int = None, level: int = None, model: PsychoacousticModel = None):
        be = not little_endian
        endian = be and '>' or '<'
        data = np.pad(data, ((0, -len(data[:, 0])%2), (0, 0)), mode='constant')
        fft_data = [mdct(data[:, i], N=len(data)*2) for i in range(channels)]

        if lossy:
            fft_data = np.sign(fft_data) * np.abs(fft_data)**(3/4)
            nfilts = len(data) // 8
            frame_size, alpha = len(data), (800 - (1.2**level))*0.001
            M = model.get_model(nfilts, frame_size, alpha, sample_rate)
            for c in range(channels):
                # fft_data[c] = np.around(fft_data[c] / 0.125) * 0.125
                mXbark = psycho.mapping2bark(np.abs(fft_data[c]),M['W'],frame_size*2)
                mTbark = psycho.maskingThresholdBark(mXbark,M['sprfuncmat'],alpha,sample_rate,nfilts) * np.log2(level+1)/2
                thres =  psycho.mappingfrombark(mTbark,M['W_inv'],frame_size*2)[:-1]
                fft_data[c][nfilts:][np.abs(fft_data[c][nfilts:]) < thres[nfilts:]] = 0

        while any(np.max(np.abs(c)) > np.finfo(fourier.dtypes[bits]).max for c in fft_data):
            if bits == 128: raise Exception('Overflow with reaching the max bit depth.')
            bits = {16:24, 24:32, 32:48, 48:64, 64:128}.get(bits, 128) 

        data: bytes = np.column_stack([d.astype(fourier.dtypes[bits]).newbyteorder(endian) for d in fft_data]).ravel(order='C').tobytes()
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
            if bits == 48: data = b''.join([be and data[i:i+6]+b'\x00\x00' or b'\x00\x00'+data[i:i+6] for i in range(0, len(data), 6)])
            elif bits == 24: data = b''.join([be and data[i:i+3]+b'\x00' or b'\x00'+data[i:i+3] for i in range(0, len(data), 3)])
        elif bits == 12:
            data = data.hex()
            if endian == '<': data = ''.join([data[i:i+3][0] + '0' + data[i:i+3][1:] for i in range(0, len(data), 3)])
            else: data = ''.join([data[i:i+3] + '0' for i in range(0, len(data), 3)])
            data = bytes.fromhex(data)
        else:
            raise Exception('Illegal bits value.')
        data_numpy = np.frombuffer(data, dtype=endian+fourier.dtypes[bits]).astype(float)

        freq = [data_numpy[i::channels] for i in range(channels)]
        freq = np.where(np.isnan(freq) | np.isinf(freq), 0, freq)
        if lossy: freq = np.sign(freq) * np.abs(freq)**(4/3)

        return np.column_stack([imdct(d, N=len(d)*2) for d in freq])
