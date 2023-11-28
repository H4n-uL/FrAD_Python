import hashlib
import json
from ml_dtypes import bfloat16
import numpy as np
from scipy.fft import fft
from scipy.signal import resample
import subprocess
from .tools.ecc import ecc
from .tools.headb import headb

class encode:
    def audio(data, bits: int, channels: int, osr: int, nsr: int = None):
        if nsr and nsr != osr:
            resdata = np.zeros((int(len(data) * nsr / osr), channels))
            for i in range(channels):
                resdata[:, i] = resample(data[:, i], int(len(data[:, i]) * nsr / osr))
            data = resdata

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
    
    def get_info(file_path):
        command = ['ffprobe', 
                '-v', 'quiet', 
                '-print_format', 'json', 
                '-show_streams', 
                file_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        info = json.loads(result.stdout)

        for stream in info['streams']:
            if stream['codec_type'] == 'audio':
                return int(stream['channels']), int(stream['sample_rate'])
        return None
    
    def get_pcm(file_path: str):
        command = [
            'ffmpeg',
            '-i', file_path,
            '-f', 's32le',
            '-acodec', 'pcm_s32le',
            '-vn',
            'pipe:1'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pcm_data, _ = process.communicate()
        channels, sample_rate = encode.get_info(file_path)
        data = np.frombuffer(pcm_data, dtype=np.int32).reshape(-1, channels)
        return data, sample_rate, channels

    def enc(file_path: str, bits: int, out: str = None, apply_ecc: bool = False,
                new_sample_rate: int = None, 
                meta = None, img: bytes = None):
        data, sample_rate, channel = encode.get_pcm(file_path)
        sample_rate_bytes = (new_sample_rate if new_sample_rate is not None else sample_rate).to_bytes(3, 'little')

        if data.dtype == np.uint8:
            data = (data.astype(np.int32) - 2**7) * 2**24
        elif data.dtype == np.int16:
            data = data.astype(np.int32) * 2**16
        elif data.dtype == np.int32:
            pass
        else:
            raise ValueError('Unsupported bit depth')

        data = encode.audio(data, bits, channel, sample_rate, new_sample_rate)

        # if channel == 1:
        #     data = encode.mono(data, bits, sample_rate, new_sample_rate)
        # elif channel == 2:
        #     data = encode.stereo(data, bits, sample_rate, new_sample_rate)
        # else:
        #     raise Exception('Fourier Analogue only supports Mono and Stereo.')

        data = ecc.encode(data, apply_ecc)
        checksum = hashlib.md5(data).digest()

        h = headb.uilder(sample_rate_bytes, channel=channel, bits=bits, isecc=apply_ecc, md5=checksum,
            meta=meta, img=img)

        if not (out.endswith('.fra') or out.endswith('.fva') or out.endswith('.sine')):
            out += '.fra'

        with open(out if out is not None else'fourierAnalogue.fra', 'wb') as f:
            f.write(h)
            f.write(data)
