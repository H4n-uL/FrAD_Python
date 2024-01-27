from .common import variables
from .fourier import fourier
import hashlib, json, os, shutil, struct, subprocess, sys, time, zlib
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
                return int(stream['channels']), int(stream['sample_rate']), stream['codec_name']
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
        metadata_lines = meta.split("\n")[1:]
        metadata = []
        current_key = None
        current_value = []

        for line in metadata_lines:
            if "=" in line:
                if current_key:
                    metadata.append([current_key, "\n".join(current_value).replace("\n\\\n", "\n")])
                current_key, value = line.split("=", 1)
                if current_key in excluded:
                    current_key = None
                else:
                    current_value = [value]
            elif current_key:
                current_value.append(line)

        if current_key:
            metadata.append([current_key, "\n".join(current_value)])
        os.remove(variables.meta)
        return metadata

    def get_image(file_path: str):
        command = [
            variables.ffmpeg, '-v', 'quiet', '-i', file_path, 
            '-an', '-vcodec', 'copy', 
            '-f', 'image2pipe', '-'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        image, _ = process.communicate()
        return image

    def enc(file_path: str, bits: int,
                out: str = None, apply_ecc: bool = False,
                meta = None, img: bytes = None,
                verbose: bool = False):
        nperseg = 2048
        ecc_dsize = 128
        ecc_codesize = 20

        # Getting Audio info w. ffmpeg & ffprobe
        channels, sample_rate, codec = encode.get_info(file_path)
        segmax = (2**32 // (((ecc_dsize+ecc_codesize)/ecc_dsize if apply_ecc else 1) * channels * bits // 8)//4)*4
        if nperseg > segmax: raise ValueError(f'Sample size cannot exceed {segmax}.')
        if nperseg < 4: raise ValueError(f'Sample size must be at least 4.')
        if nperseg % 4 != 0: raise ValueError('Sample size must be multiple of 4.')

        encode.get_pcm(file_path)

        # Fourier Transform
        try:
            start_time = time.time()
            total_bytes = 0
            cli_width = 40
            sample_size = bits // 4 * channels
            dlen = os.path.getsize(variables.temp_pcm)

            with open(variables.temp_pcm, 'rb') as pcm:
              with open(variables.temp, 'wb') as swv:
                if verbose: print('\n')
                while True:
                    p = pcm.read(nperseg * 4 * channels)                           # Reading PCM
                    if not p: break                                                # if no data, Break
                    block = np.frombuffer(p, dtype=np.int32).reshape(-1, channels) # RAW PCM to Numpy
                    segment = fourier.analogue(block, bits, channels)              # Fourier Transform

                    # Applying ECC (This will make encoding thousands of times slower)
                    if apply_ecc: segment = ecc.encode(segment, ecc_dsize, ecc_codesize)
                    else: ecc_dsize = ecc_codesize = 0

                    # block = zlib.compress(block)

                    # WRITE
                    swv.write(
                        b'\xff\xd0\xd2\x97' + \
                        struct.pack('>I', len(segment)) + \
                        headb.encode_efb(apply_ecc, bits) + \
                        struct.pack('>B', channels - 1) + \
                        struct.pack('>B', ecc_dsize) + \
                        struct.pack('>B', ecc_codesize) + \
                        struct.pack('>I', zlib.crc32(segment)) + \

                        segment
                    )

                    if verbose:
                        total_bytes += len(block) * sample_size
                        elapsed_time = time.time() - start_time
                        bps = total_bytes / elapsed_time
                        mult = bps / sample_rate / sample_size
                        percent = total_bytes / dlen / bits * 1600
                        b = int(percent / 100 * cli_width)
                        print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                        print(f'Encode Speed: {(bps / 10**6):.3f} MB/s, X{mult:.3f}')
                        print(f"[{'█'*b}{' '*(cli_width-b)}] {percent:.3f}% completed")
                if verbose: print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
            os.remove(variables.temp_pcm)
        except KeyboardInterrupt:
            print('Aborting...')
            os.remove(variables.temp)
            os.remove(variables.temp_pcm)
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
            if img == None: img = encode.get_image(file_path)

            # Moulding header
            h = headb.uilder(sample_rate, channel=channels,
                md5=checksum,
                meta=meta, img=img)

            if out is None: out = os.path.basename(file_path).rsplit('.', 1)[0]

            # Setting file extension
            if not (out.lower().endswith('.frad') or out.lower().endswith('.dsin') or out.lower().endswith('.fra') or out.lower().endswith('.dsn')):
                if len(out) <= 8: out += '.fra'
                else: out += '.frad'

            # Creating Fourier Analogue-in-Digital File
            with open(out, 'wb') as file:
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
