from .common import variables
from .fourier import fourier
import hashlib
import json
import numpy as np
import os
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
    
    def get_metadata(file_path: str):
        excluded = ['major_brand', 'minor_version', 'compatible_brands', 'encoder']
        command = [
            variables.ffmpeg, '-v', 'quiet',
            '-i', file_path,
            '-f', 'ffmetadata',
            variables.meta
        ]
        subprocess.run(command)
        with open(variables.meta, 'r') as m:
            meta = m.read()
        metadata_lines = meta.split("\n")[1:]  # 첫 줄만 제외합니다.
        metadata = []
        current_key = None
        current_value = []

        for line in metadata_lines:
            if "=" in line:  # '='이 있는 줄이면 새로운 항목이 시작된 것입니다.
                if current_key:  # 이전에 처리하던 항목이 있으면 metadata에 추가합니다.
                    metadata.append([current_key, "\n".join(current_value).replace("\n\\\n", "\n")])
                current_key, value = line.split("=", 1)  # 새로운 항목의 키와 값을 분리합니다.
                if current_key in excluded:  # 제외할 항목이면 current_key를 None으로 설정합니다.
                    current_key = None
                else:
                    current_value = [value]
            elif current_key:  # '='이 없는 줄이면 이전 항목의 값이 계속되는 것입니다.
                current_value.append(line)

        if current_key:  # 마지막에 처리하던 항목이 있으면 metadata에 추가합니다.
            metadata.append([current_key, "\n".join(current_value)])
        os.remove(variables.meta)
        return metadata

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
        with open(variables.temp, 'wb') as temp:
            for i in range(0, len(data), nperseg):
                block = data[i:i+nperseg]
                segment = fourier.analogue(block, bits, channel)
                segment = ecc.encode(segment, apply_ecc) # Encoding Reed-Solomon ECC
                temp.write(segment)
            temp.seek(0)
        # Calculating MD5 hash
        with open(variables.temp, 'rb') as temp:
            checksum = hashlib.md5(temp.read()).digest()

        if meta == None: meta = encode.get_metadata(file_path)

        # Moulding header
        h = headb.uilder(sample_rate, channel=channel, bits=bits, isecc=apply_ecc, md5=checksum,
            meta=meta, img=img)

        # Setting file extension
        if not (out.endswith('.fra') or out.endswith('.fva') or out.endswith('.sine')):
            out += '.fra'

        # Creating Fourier Analogue-in-Digital File
        with open(out if out is not None else'fourierAnalogue.fra', 'wb') as file:
            file.write(h)
            with open(variables.temp, 'r+b') as swv:
                while True:
                    block = swv.read()
                    if block: file.write(block)
                    else: break
            os.remove(variables.temp)
