from .common import variables, methods
from .fourier import fourier
import json, os, math, random, struct, subprocess, sys, time, traceback, typing, zlib
import numpy as np
from .tools.ecc import ecc
from .tools.headb import headb

class encode:
    @staticmethod
    def overlap(data: np.ndarray, prev: np.ndarray, olap: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        if len(prev) != 0: data = np.concatenate([prev, data])
        if kwargs.get('profile') in [1, 2] and olap: prev = data[-kwargs.get('fsize', None)//olap:]
        else: prev = np.array([])
        return data, prev

    @staticmethod
    def write_frame(file: typing.BinaryIO, frame: bytes, channels: int, srate: int, pfb: bytes, ecc_list: tuple[int, int], fsize: int, **kwargs) -> None:
        profile, isecc, _, _ = headb.decode_pfb(pfb)
        if not isecc: ecc_list = (0, 0)
        data = bytes(
            variables.FRM_SIGN +
            struct.pack('>I', len(frame)) +
            pfb
        )
        if profile == 0:
            data += (
                struct.pack('>B', channels - 1) +
                struct.pack('>B', ecc_list[0]) +
                struct.pack('>B', ecc_list[1]) +
                struct.pack('>I', srate) +
                b'\x00'*8 +
                struct.pack('>I', fsize) +
                struct.pack('>I', zlib.crc32(frame))
            )
        elif profile == 1:
            data += (
                headb.encode_css_prf1(channels, srate, fsize) +
                struct.pack('>B', kwargs.get('olap', 0))
            )
            if isecc:
                data += (
                    struct.pack('>B', ecc_list[0]) +
                    struct.pack('>B', ecc_list[1]) +
                    struct.pack('>H', methods.crc16_ansi(frame))
                )
        data += frame
        file.write(data)
        return None

    @staticmethod
    def get_info(file_path) -> tuple[int, int, str, int]:
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
                duration = stream['duration_ts'] * int(stream['sample_rate']) // int(stream['time_base'][2:])
                return int(stream['channels']), int(stream['sample_rate']), stream['codec_name'], duration
        print('No audio stream found.')
        sys.exit(1)

    @staticmethod
    def get_pcm_command(file_path: str, osr: int, new_srate: int | None, chnl: int | None) -> list[str]:
        command = [
            variables.ffmpeg,
            '-v', 'quiet',
            '-i', file_path,
            '-f', 'f64be',
            '-vn'
        ]
        if new_srate is not None and new_srate != osr:
            command.extend(['-ar', str(new_srate)])
        if chnl is not None:
            command.extend(['-ac', str(chnl)])
        command.append('pipe:1')
        return command

    @staticmethod
    def get_metadata(file_path: str):
        excluded = ['major_brand', 'minor_version', 'compatible_brands', 'encoder']
        command = [
            variables.ffmpeg, '-v', 'quiet', '-y',
            '-i', file_path,
            '-f', 'ffmetadata',
            variables.meta
        ]
        subprocess.run(command)
        with open(variables.meta, 'r', encoding='utf-8') as m:
            meta = m.read()
        metadata_lines = meta.split('\n')[1:]
        metadata = []
        current_key = None
        current_value = []

        for line in metadata_lines:
            if '=' in line:
                if current_key:
                    metadata.append([current_key, '\n'.join(current_value).replace('\n\\\n', '\n')])
                current_key, value = line.split('=', 1)
                if current_key in excluded:
                    current_key = None
                else:
                    current_value = [value]
            elif current_key:
                current_value.append(line)

        if current_key:
            metadata.append([current_key, '\n'.join(current_value)])
        os.remove(variables.meta)
        return metadata

    @staticmethod
    def get_image(file_path: str):
        command = [
            variables.ffmpeg, '-v', 'quiet', '-i', file_path,
            '-an', '-vcodec', 'copy',
            '-f', 'image2pipe', '-'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        image, _ = process.communicate()
        return image

    @staticmethod
    def enc(file_path: str, bits: int, **kwargs):
        # FrAD data specification
        fsize: int = kwargs.get('fsize', 2048)
        little_endian: bool = kwargs.get('le', False)
        profile: int = kwargs.get('prf', 0)
        loss_level: int = kwargs.get('lv', 0)
        overlap: int = kwargs.get('olap', variables.overlap_rate)
        gain: float = kwargs.get('gain', None)

        # ECC settings
        apply_ecc: bool = kwargs.get('ecc', False)
        ecc_sizes: tuple[int, int] = kwargs.get('ecc_sizes', [128, 20])
        ecc_dsize: int = ecc_sizes[0]
        ecc_codesize: int = ecc_sizes[1]

        # Audio settings
        new_srate: int = kwargs.get('srate', None)
        chnl: int = kwargs.get('chnl', None)

        # Raw PCM
        raw: str = kwargs.get('raw', None)

        # Metadata
        meta: list[list[str]] = kwargs.get('meta', None)
        img: bytes = kwargs.get('img', None)

        # CLI
        verbose: bool = kwargs.get('verbose', False)
        out: str = kwargs.get('out', None)

        channels: int = 0
        smprate: int = 0

# --------------------------- Pre-Encode error checks ---------------------------- #
        methods.cantreencode(open(file_path, 'rb').read(4))

        # Forcing sample rate and channel count for raw PCM
        if raw:
            if new_srate is None: print('Sample rate is required for raw PCM.'); sys.exit(1)
            if chnl is None: print('Channel count is required for raw PCM.'); sys.exit(1)
            channels, smprate = chnl, new_srate
        if not 20 >= loss_level >= 0: print(f'Invalid compression level: {loss_level} Lossy compression level should be between 0 and 20.'); sys.exit(1)

# ------------------------------ Pre-Encode settings ----------------------------- #
        # Getting Audio info w. ffmpeg & ffprobe
        duration = 0
        cmd = []
        if not raw:
            channels, smprate, codec, duration = encode.get_info(file_path)
            if profile in [1, 2]:
                new_srate = min(new_srate or smprate, 96000)
                if not new_srate in variables.prf1_srates: new_srate = 48000
            cmd = encode.get_pcm_command(file_path, smprate, new_srate, chnl)
            if chnl is not None: channels = chnl
            # segmax for Profile 0 = 4GiB / (intra-channel-sample size * channels * ECC mapping)
            # intra-channel-sample size = bit depth * 8, least 3 bytes(float s1e8m15)
            # ECC mapping = (block size / data size)
            segmax = {0: (2**32-1) // (((ecc_dsize+ecc_codesize)/ecc_dsize if apply_ecc else 1) * channels * max(bits/8, 3)),
                      1: max(variables.prf1_smpls_li)}
            if fsize > segmax[profile]: print(f'Sample size cannot exceed {segmax}.'); sys.exit(1)
            if profile == 1: fsize = min((x for x in variables.prf1_smpls_li if x >= fsize), default=None)
            if new_srate is not None: duration = int(duration / smprate * new_srate)
            if meta == None: meta = encode.get_metadata(file_path)
            if img  == None: img  = encode.get_image(   file_path)

        smprate = new_srate is not None and new_srate or smprate

        if overlap <= 0: overlap = 0
        else:
            if overlap < 2: overlap = 1/overlap
            if overlap%1!=0: overlap = int(overlap)
            if overlap > 255: overlap = 255
        # Setting file extension
        if out is None: out = os.path.basename(file_path).rsplit('.', 1)[0]
        if not out.lower().endswith(('.frad', '.dsin', '.fra', '.dsn')):
            if profile == 0:
                if len(out) <= 8 and all(ord(c) < 128 for c in out): out += '.fra'
                else: out += '.frad'
            else:
                if len(out) <= 8 and all(ord(c) < 128 for c in out): out += '.dsn'
                else: out += '.dsin'

        if os.path.exists(out):
            print(f'{out} Already exists. Proceed?')
            while True:
                x = input('> ').lower()
                if x == 'y': break
                if x == 'n': sys.exit('Aborted.')

# ----------------------------------- Encoding ----------------------------------- #
        try:
            start_time = time.time()
            total_bytes, total_samples = 0, 0
            cli_width = 40

            prev = np.array([])
            dtype, sample_bytes = methods.get_dtype(raw)
            smpsize = sample_bytes * channels # Single sample size = bit depth * channels

            # Open FFmpeg
            if not raw: process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            else:
                rfile = open(file_path, 'rb')
                duration = os.path.getsize(file_path) / smpsize

            printed = False
            # Write file
            open(out, 'wb').write(headb.uilder(meta, img))
            with open(out, 'ab') as file:
                while True:
                    # bits = random.choice([12, 16, 24, 32, 48, 64]) # Random bit depth test
                    # fsize = random.choice(list(range(32, 8193))) # Random spf test
                    # profile = random.choice(list(range(2))) # Random profile test
                    # loss_level = random.choice(list(range(21))) # Random lossy level test
                    # apply_ecc = random.choice([True, False]) # Random ECC test
                    # ecc_dsize, ecc_codesize = random.choice(list(range(64, 129))), random.choice(list(range(16, 64))) # Random ECC test

                    # Getting required read length
                    rlen = fsize
                    while rlen < len(prev): rlen += 128
                    # Overlap
                    if profile in [1, 2] and len(prev) != 0: rlen -= len(prev)

                    if not raw:
                        if process.stdout is None: raise FileNotFoundError('Broken pipe.')
                        data = process.stdout.read(rlen * smpsize) # Reading PCM
                    else: data = rfile.read(rlen * smpsize)        # Reading RAW PCM
                    if not data: break                             # if no data, Break

                    # RAW PCM to Numpy
                    frame = np.frombuffer(data[:len(data)//smpsize * smpsize], dtype).astype(float).reshape(-1, channels) * gain
                    rlen = len(frame)
                    frame, prev = encode.overlap(frame, prev, overlap, fsize=fsize, chnl=channels, profile=profile)
                    if raw:
                        if not raw.startswith('f'):
                            frame /= 2**(sample_bytes*8-1)
                            if raw.startswith('u'): frame-=1
                    flen = len(frame)

                    # Encoding
                    frame, bit_depth_frame, channels_frame, bits_pfb = \
                        fourier.analogue(frame, bits, channels, little_endian, profile=profile, smprate=smprate, level=loss_level)

                    # Applying ECC
                    if apply_ecc: frame = ecc.encode(frame, ecc_dsize, ecc_codesize)

                    # EFloat Byte
                    pfb = headb.encode_pfb(profile, apply_ecc, little_endian, bits_pfb)
                    encode.write_frame(file, frame, channels_frame, smprate, pfb, (ecc_dsize, ecc_codesize), flen, olap=overlap)

                    # Verbose block
                    if verbose:
                        sample_size = bit_depth_frame // 8 * channels
                        total_bytes += rlen * sample_size
                        total_samples += rlen
                        elapsed_time = time.time() - start_time
                        bps = total_bytes / elapsed_time
                        mult = bps / smprate / sample_size
                        printed = methods.logging(3, 'Encode', printed, percent=(total_samples/duration*100), bps=bps, mult=mult, time=elapsed_time)

        except KeyboardInterrupt:
            print('Aborting...')
            sys.exit(0)
        except Exception as e:
            if verbose: print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
            sys.exit(traceback.format_exc())
