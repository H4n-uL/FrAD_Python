from .common import variables
from .fourier import fourier
import hashlib
import json
import numpy as np
from scipy.signal import resample
import subprocess
from .tools.ecc import ecc
from .tools.headb import headb

class encode:
    def get_info(file_path):
        command = [variables.ffprobe,
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
            variables.ffmpeg,
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
        # Getting Audio info w. ffmpeg & ffprobe
        data, sample_rate, channel = encode.get_pcm(file_path)

        # Resampling
        if new_sample_rate:
            resdata = np.zeros((int(len(data) * new_sample_rate / sample_rate), channel))
            for i in range(channel):
                resdata[:, i] = resample(data[:, i], int(len(data[:, i]) * new_sample_rate / sample_rate))
            data = resdata

        # Applying Sample rate
        sample_rate = (new_sample_rate if new_sample_rate is not None else sample_rate)

        # Fourier Transform
        nperseg = variables.nperseg
        fft_data = []
        for i in range(0, len(data), nperseg):
            block = data[i:i+nperseg]
            segment = fourier.analogue(block, bits, channel)
            fft_data.append(segment)
        data = b''.join(fft_data)

        # Encoding Reed-Solomon ECC
        data = ecc.encode(data, apply_ecc)
        # Calculating MD5 hash
        checksum = hashlib.md5(data).digest()

        # Moulding header
        h = headb.uilder(sample_rate, channel=channel, bits=bits, isecc=apply_ecc, md5=checksum,
            meta=meta, img=img)

        # Setting file extension
        if not (out.endswith('.fra') or out.endswith('.fva') or out.endswith('.sine')):
            out += '.fra'

        # Merger
        file = h + data

        # Creating Fourier Analogue-in-Digital File
        with open(out if out is not None else'fourierAnalogue.fra', 'wb') as f:
            f.write(file)
