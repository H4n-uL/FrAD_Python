from .common import variables
from .fourier import fourier
import hashlib, json, os, shutil, subprocess, sys
import numpy as np
from scipy.signal import resample
from .tools.ecc import ecc
from .tools.headb import headb

class encode:
    def get_info(file_path):
        command = [variables.ffprobe,
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            file_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        info = json.loads(result.stdout)

        for stream in info['streams']:
            if stream['codec_type'] == 'audio':
                return int(stream['channels']), int(stream['sample_rate'])
        return None

    def get_pcm(file_path: str):
        command = [
            variables.ffmpeg,
            '-v', 'quiet',
            '-i', file_path,
            '-f', 's32le',
            '-acodec', 'pcm_s32le',
            '-vn',
            variables.temp_pcm
        ]
        subprocess.run(command)
        channels, sample_rate = encode.get_info(file_path)
        return sample_rate, channels
    
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
        sample_rate, channel = encode.get_pcm(file_path)
        try:
            # Resampling
            if new_sample_rate:
              try:
                with open(variables.temp_pcm, 'rb') as pcmb:
                  with open(variables.temp2_pcm, 'wb') as pcma:
                    while True:
                        wv = pcmb.read(sample_rate * channel * 4)
                        if not wv: break
                        block = np.frombuffer(wv, dtype=np.int32).reshape(-1, channel)
                        resdata = np.zeros((new_sample_rate, channel))
                        for i in range(channel):
                            resdata[:, i] = resample(block[:, i], new_sample_rate)
                        pcma.write(resdata.astype(np.int32).tobytes())
                shutil.move(variables.temp2_pcm, variables.temp_pcm)
              except KeyboardInterrupt:
                os.remove(variables.temp_pcm)
                os.remove(variables.temp2_pcm)
                sys.exit(1)

            # Applying Sample rate
            sample_rate = (new_sample_rate if new_sample_rate is not None else sample_rate)

            # Fourier Transform
            nperseg = variables.nperseg
        except KeyboardInterrupt: 
            print('Aborting...')
            os.remove(variables.temp_pcm)
            sys.exit(1)

        try:
            with open(variables.temp_pcm, 'rb') as pcm:
              with open(variables.temp, 'wb') as swv:
                while True:
                    p = pcm.read(nperseg * 4 * channel)
                    if not p: break
                    block = np.frombuffer(p, dtype=np.int32).reshape(-1, channel)
                    segment = fourier.analogue(block, bits, channel)
                    swv.write(segment)
                os.remove(variables.temp_pcm)
        except KeyboardInterrupt:
            print('Aborting...')
            os.remove(variables.temp)
            os.remove(variables.temp_pcm)
            sys.exit(1)

        try:
            with open(variables.temp, 'rb') as swv:
              with open(variables.temp2, 'wb') as enf:
                while True:
                    block = swv.read(16777216)
                    if not block: break
                    segment = ecc.encode(block, apply_ecc) # Encoding Reed-Solomon ECC
                    enf.write(segment)
                shutil.move(variables.temp2, variables.temp)
        except KeyboardInterrupt:
            print('Aborting...')
            os.remove(variables.temp2)
            os.remove(variables.temp)
            sys.exit(1)

        try:
            # Calculating MD5 hash
            with open(variables.temp, 'rb') as temp:
                md5 = hashlib.md5()
                while True:
                    d = temp.read(variables.hash_block_size)
                    if not d: break
                    md5.update(d)
                checksum = md5.digest()

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
        except KeyboardInterrupt:
            print('Aborting...')
            os.remove(variables.temp)
            sys.exit(1)
