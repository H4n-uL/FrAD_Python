from .ffpath import ff
from .fourier import fourier
import hashlib
import json
import numpy as np
import subprocess
from .tools.ecc import ecc
from .tools.headb import headb

class encode:
    def get_info(file_path):
        command = [ff.probe,
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
            ff.mpeg,
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

        data = fourier.analogue(data, bits, channel, sample_rate, new_sample_rate)
        data = ecc.encode(data, apply_ecc)
        checksum = hashlib.md5(data).digest()

        h = headb.uilder(sample_rate_bytes, channel=channel, bits=bits, isecc=apply_ecc, md5=checksum,
            meta=meta, img=img)

        if not (out.endswith('.fra') or out.endswith('.fva') or out.endswith('.sine')):
            out += '.fra'

        with open(out if out is not None else'fourierAnalogue.fra', 'wb') as f:
            f.write(h)
            f.write(data)
