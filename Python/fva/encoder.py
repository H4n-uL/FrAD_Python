from .tools.ecc import ecc
from ml_dtypes import bfloat16
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import resample
from .tools.header import header

class encode:
    def mono(data, bits: int, osr: int, nsr: int = None):
        if nsr and nsr != osr:
            resdata = np.zeros(int(len(data) * nsr / osr))
            resdata = resample(data, int(len(data) * nsr / osr))
            data = resdata

        fft_data = fft(data)

        # if bits == 512:
        #     amp = np.abs(fft_data).astype(np.float512); pha = np.angle(fft_data).astype(np.float512)
        # elif bits == 256:
        #     amp = np.abs(fft_data).astype(np.float256); pha = np.angle(fft_data).astype(np.float256)
        # elif bits == 128:
        #     amp = np.abs(fft_data).astype(np.float128); pha = np.angle(fft_data).astype(np.float128)
        if bits == 64:
            amp = np.abs(fft_data).astype(np.float64); pha = np.angle(fft_data).astype(np.float64)
        elif bits == 32:
            amp = np.abs(fft_data).astype(np.float32); pha = np.angle(fft_data).astype(np.float32)
        elif bits == 16:
            amp = np.abs(fft_data).astype(bfloat16); pha = np.angle(fft_data).astype(bfloat16)
        else:
            raise Exception('Illegal bits value.')

        data = np.column_stack((amp, pha)).tobytes()
        return data

    def stereo(data, bits: int, osr: int, nsr: int = None):
        if nsr and nsr != osr:
            resdata = np.zeros((int(len(data) * nsr / osr), 2))
            resdata[:, 0] = resample(data[:, 0], int(len(data[:, 0]) * nsr / osr))
            resdata[:, 1] = resample(data[:, 1], int(len(data[:, 1]) * nsr / osr))
            data = resdata

        chleft  = data[:, 0]
        chright = data[:, 1]
        fftleft = fft(chleft); fftright = fft(chright)

        # if bits == 512:
        #     ampleft  = np.abs(fftleft) .astype(np.float512); phaleft  = np.angle(fftleft) .astype(np.float512)
        #     ampright = np.abs(fftright).astype(np.float512); pharight = np.angle(fftright).astype(np.float512)
        # elif bits == 256:
        #     ampleft  = np.abs(fftleft) .astype(np.float256); phaleft  = np.angle(fftleft) .astype(np.float256)
        #     ampright = np.abs(fftright).astype(np.float256); pharight = np.angle(fftright).astype(np.float256)
        # elif bits == 128:
        #     ampleft  = np.abs(fftleft) .astype(np.float128); phaleft  = np.angle(fftleft) .astype(np.float128)
        #     ampright = np.abs(fftright).astype(np.float128); pharight = np.angle(fftright).astype(np.float128)
        if bits == 64:
            ampleft  = np.abs(fftleft) .astype(np.float64); phaleft  = np.angle(fftleft) .astype(np.float64)
            ampright = np.abs(fftright).astype(np.float64); pharight = np.angle(fftright).astype(np.float64)
        elif bits == 32:
            ampleft  = np.abs(fftleft) .astype(np.float32); phaleft  = np.angle(fftleft) .astype(np.float32)
            ampright = np.abs(fftright).astype(np.float32); pharight = np.angle(fftright).astype(np.float32)
        elif bits == 16:
            ampleft  = np.abs(fftleft) .astype(bfloat16); phaleft  = np.angle(fftleft) .astype(bfloat16)
            ampright = np.abs(fftright).astype(bfloat16); pharight = np.angle(fftright).astype(bfloat16)
        else:
            raise Exception('Illegal bits value.')

        data_left  = np.column_stack((ampleft,  phaleft))
        data_right = np.column_stack((ampright, pharight))

        data = np.ravel(np.column_stack((data_left, data_right)), order='C').tobytes()
        return data

    def enc(filename, bits: int, out: str = None, apply_ecc: bool = False,
                new_sample_rate: int = None, title: str = None, artist: str = None,
                lyrics: str = None, album: str = None, track_number: int = None,
                genre: str = None, date: str = None, description: str = None,
                comment: str = None, composer: str = None, copyright: str = None,
                license: str = None, organization: str = None, location: str = None,
                performer: str = None, isrc: str = None, img: bytes = None):
        sample_rate, data = wavfile.read(filename)
        sample_rate_bytes = (new_sample_rate if new_sample_rate is not None else sample_rate).to_bytes(3, 'little')

        if data.dtype == np.uint8:
            data = (data.astype(np.int32) - 2**7) * 2**24
        elif data.dtype == np.int16:
            data = data.astype(np.int32) * 2**16
        elif data.dtype == np.int32:
            pass
        else:
            raise ValueError('Unsupported WAV bit depth')

        channel = len(data.shape)

        if len(data.shape) == 1:
            data = encode.mono(data, bits, sample_rate, new_sample_rate)
        elif len(data.shape) == 2:
            data = encode.stereo(data, bits, sample_rate, new_sample_rate)
        else:
            raise Exception('Fourier Analogue only supports Mono and Stereo.')

        data = ecc.encode(data, apply_ecc)

        h = header.builder(sample_rate_bytes, channel=channel, bits=bits, isecc=apply_ecc,
            title=title, lyrics=lyrics, artist=artist, album=album,
            track_number=track_number,
            genre=genre, date=date,
            description=description,
            comment=comment, composer=composer,
            copyright=copyright, license=license,
            organization=organization, location=location,
            performer=performer, isrc=isrc, img=img)

        if out is not None and out[-4:-1]+out[-1] != '.fva':
            out += '.fva'

        with open(out if out is not None else'fourierAnalogue.fva', 'wb') as f:
            f.write(h)
            f.write(data)
