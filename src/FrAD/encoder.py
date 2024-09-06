from .common import variables, methods, terminal
from .profiles.prf import profiles, compact
from .fourier import fourier
import io, json, os, math, random, struct, subprocess, sys, time, traceback, zlib
import numpy as np
from .tools.asfh import ASFH
from .tools.ecc import ecc
from .tools.headb import headb

class encode:
    @staticmethod
    def overlap(frame: np.ndarray, overlap_fragment: np.ndarray, olap: int, profile: int) -> tuple[np.ndarray, np.ndarray]:
        olap = olap > 1 and min(max(olap, 2), 256) or 0
        if overlap_fragment.shape != np.array([]).shape: frame = np.concatenate([overlap_fragment, frame])
        next_overlap = np.array([])
        if profile in profiles.COMPACT and olap: next_overlap = frame[(len(frame) * (olap - 1)) // olap:]
        return frame, next_overlap

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
                tbase = stream['time_base'].split('/')
                duration = stream['duration_ts'] * int(stream['sample_rate']) // int(tbase[1]) * int(tbase[0])
                return int(stream['channels']), int(stream['sample_rate']), stream['codec_name'], duration
        terminal('No audio stream found.')
        sys.exit(1)

    @staticmethod
    def get_pcm_command(file_path: str, raw: tuple[str, int | None, int | None], srate: int | None, chnl: int | None) -> list[str]:
        command = [
            variables.ffmpeg,
            '-v', 'quiet']

        if raw: command.extend([
            '-f', raw[0],
            '-ar', str(raw[1]),
            '-ac', str(raw[2])])

        command.extend([
            '-i', file_path,
            '-f', 'f64be',
            '-vn'])

        if srate is not None:
            command.extend(['-ar', str(srate)])
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
        new_chnl: int = kwargs.get('chnl', None)

        # Raw PCM
        raw: tuple[str, int, int] = kwargs.get('raw', ('f64be', 0, 0))
        # raw: str = kwargs.get('raw', None)

        # Metadata
        meta: list[list[str]] = kwargs.get('meta', None)
        img: bytes = kwargs.get('img', None)

        # CLI
        verbose: bool = kwargs.get('verbose', False)
        out: str = kwargs.get('out', None)

        channels: int = 0
        srate: int = 0

# --------------------------- Pre-Encode error checks ---------------------------- #
        methods.cantreencode(open(file_path, 'rb').read(4))

        try: variables.bit_depths[profile].index(bits)
        except: terminal(f'Invalid bit depth {bits} for Profile {profile}'); sys.exit(1)
        if not 20 >= loss_level >= 0: terminal(f'Invalid compression level: {loss_level} Lossy compression level should be between 0 and 20.'); sys.exit(1)

        if fsize > variables.segmax[profile]: terminal(f'Sample size cannot exceed {variables.segmax[profile]}.'); sys.exit(1)

# ------------------------------ Pre-Encode settings ----------------------------- #
        duration = 0

        # File info and metadata
        # original channels and srate
        if not raw:
            channels, srate, codec, duration = encode.get_info(file_path)
            if new_srate is not None: duration = int(duration / srate * new_srate)
            if meta == None: meta = encode.get_metadata(file_path)
            if img  == None: img  = encode.get_image(   file_path)
        else:
            channels, srate = raw[2], raw[1]
            duration = os.path.getsize(file_path) / methods.get_dtype(raw[0])[1] * ((new_srate or srate) / srate) / channels

        # Modifying new_srate for Profile 1
        if profile in profiles.COMPACT:
            new_srate = min(new_srate or srate, 96000)
            if not new_srate in compact.srates: new_srate = 48000
            fsize = min((x for x in compact.samples_li if x >= fsize), default=2048)

        # Moulding FFmpeg command and initting read srates and channels
        cmd = encode.get_pcm_command(file_path, raw, new_srate, new_chnl)
        srate, channels = new_srate or srate, new_chnl or channels

        if not isinstance(overlap, (int, float)): overlap = variables.overlap_rate
        elif overlap <= 0: overlap = 0
        elif overlap <= 0.5: overlap = int(1/overlap)
        elif overlap < 2: overlap = 2
        elif overlap > 255: overlap = 255
        if overlap%1!=0: overlap = int(overlap)
        # Setting file extension
        if out is None: out = os.path.basename(file_path).rsplit('.', 1)[0]
        if not out.lower().endswith(('.frad', '.dsin', '.fra', '.dsn')):
            if profile in profiles.LOSSLESS:
                if len(out) <= 8 and all(ord(c) < 128 for c in out): out += '.fra'
                else: out += '.frad'
            else:
                if len(out) <= 8 and all(ord(c) < 128 for c in out): out += '.dsn'
                else: out += '.dsin'

        if os.path.exists(out):
            terminal(f'{out} Already exists. Proceed? (Y/N)')
            while True:
                terminal('> ', end='')
                x = input().lower()
                if x == 'y': break
                if x == 'n': sys.exit('Aborted.')

# ----------------------------------- Encoding ----------------------------------- #
        try:
            start_time = time.time()
            total_bytes, total_samples = 0, 0
            overlap_fragment = np.array([])
            asfh = ASFH()

            # Open FFmpeg
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

            printed = False
            # Write file
            open(out, 'wb').write(headb.uilder(meta, img))
            with open(out, 'ab') as file:
                while True:
                    # profile = random.choice([0, 1, 4]) # Random profile test
                    # bits = random.choice(variables.bit_depths[profile]) # Random bit depth test
                    # fsize = random.choice(variables.p1.smpls_li) # Random spf test
                    # loss_level = random.choice(range(21)) # Random lossy level test
                    # apply_ecc = random.choice([True, False]) # Random ECC test
                    # ecc_codesize = random.randrange(1, 254)
                    # ecc_dsize = random.randrange(1, 255-ecc_codesize)
                    # overlap = random.choice(range(2, 256)) # Random overlap test

                    # Getting required read length
                    rlen = fsize
                    if profile in profiles.COMPACT:
                        rlen = min((x-len(overlap_fragment) for x in compact.samples_li if x >= fsize))
                        if rlen <= 0: rlen = min((x-len(overlap_fragment) for x in compact.samples_li if x-len(overlap_fragment) >= fsize))

                    if process.stdout is None: raise FileNotFoundError('Broken pipe.')
                    data = process.stdout.read(rlen * 8 * channels) # Reading PCM
                    if not data: break                        # if no data, Break

                    # RAW PCM to Numpy
                    frame = np.frombuffer(data, '>f8').astype(float).reshape(-1, channels) * gain
                    rlen = len(frame)
                    frame, overlap_fragment = encode.overlap(frame, overlap_fragment, overlap, profile)
                    flen = len(frame)

                    # Encoding
                    frame, bits_pfb, channels_frame = \
                        fourier.analogue(frame, bits, channels, little_endian, profile=profile, srate=srate, level=loss_level)

                    # Applying ECC
                    if apply_ecc: frame = ecc.encode(frame, ecc_dsize, ecc_codesize)

                    # EFloat Byte
                    pfb = headb.encode_pfb(profile, apply_ecc, little_endian, bits_pfb)

                    asfh.float_bits, asfh.chnl, asfh.endian, asfh.profile = bits_pfb, channels_frame, little_endian, profile
                    asfh.srate, asfh.fsize, asfh.overlap = srate, fsize, overlap
                    asfh.ecc, asfh.ecc_dsize, ecc_codesize = apply_ecc, ecc_dsize, ecc_codesize

                    frame_length_tot = asfh.write_frame(file, frame)

                    # Verbose block
                    total_bytes += frame_length_tot
                    if verbose:
                        total_samples += rlen
                        elapsed_time = time.time() - start_time
                        bps = total_bytes / elapsed_time
                        mult = total_samples / elapsed_time / srate
                        printed = methods.logging(3, 'Encode', printed, percent=(total_samples/duration*100), tbytes=total_bytes, bps=bps, mult=mult, time=elapsed_time)

            process.terminate()
            bps = total_bytes / (duration / srate) * 8
            bps_log = int(math.log(bps, 1000))
            terminal(f'Estimated bitrate: {bps/10**(3*bps_log):.3f} {['', 'k', 'M', 'G', 'T'][bps_log]}bps')
        except KeyboardInterrupt:
            terminal('Aborting...')
            sys.exit(0)
        except Exception as e:
            if verbose: terminal('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
            sys.exit(traceback.format_exc())
