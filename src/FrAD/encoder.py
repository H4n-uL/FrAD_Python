from turtle import st
from .common import variables, methods
from .fourier import fourier
import json, os, math, random, struct, subprocess, sys, time, traceback, zlib
import numpy as np
from .tools.ecc import ecc
from .tools.headb import headb

class encode:
    @staticmethod
    def get_info(file_path) -> tuple[int, int, str, int] | None:
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
                duration = stream['duration_ts'] * int(stream['smprate']) // int(stream['time_base'][2:])
                return int(stream['channels']), int(stream['smprate']), stream['codec_name'], duration
        return None

    @staticmethod
    def get_pcm_command(file_path: str, osr: int, new_srate: int | None) -> list[str]:
        command = [
            variables.ffmpeg,
            '-v', 'quiet',
            '-i', file_path,
            '-f', 'f64be',
            '-vn'
        ]
        if new_srate is not None or new_srate != osr:
            command.extend(['-ar', str(new_srate)])
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
    def enc(file_path: str, bits: int, little_endian: bool = False,
                out: str | None = None, profile: int = 0, loss_level: int = 0,
                samples_per_frame: int = 2048, gain: list | None = None,
                apply_ecc: bool = False, ecc_sizes: list = ['128', '20'],
                new_srate: int | None = None,
                meta = None, img: bytes | None = None,
                verbose: bool = False):
        ecc_dsize = int(ecc_sizes[0])
        ecc_codesize = int(ecc_sizes[1])

        methods.cantreencode(open(file_path, 'rb').read(4))

        if not 20 >= loss_level >= 0: raise ValueError(f'Lossy compression level should be between 0 and 20.')
        if profile in [1] and 'y' not in input('\033[1m!!!Warning!!!\033[0m\nFourier Analogue-in-Digital is designed to be an uncompressed archival codec. Compression increases the difficulty of decoding and makes data very fragile, making any minor damage likely to destroy the entire frame. Proceed? (Y/N) ').lower(): sys.exit('Aborted.')

        # Getting Audio info w. ffmpeg & ffprobe
        streaminfo = encode.get_info(file_path)
        if type(streaminfo)==tuple: channels, smprate, codec, duration = streaminfo
        else: raise ValueError('No audio stream found.')
        segmax = (2**31-1) // (((ecc_dsize+ecc_codesize)/ecc_dsize if apply_ecc else 1) * channels * 16)//16
        if samples_per_frame > segmax: raise ValueError(f'Sample size cannot exceed {segmax}.')
        if bits == 12 and samples_per_frame % 2 != 0: raise ValueError(f'Samples per frame should be even for 12-bit encoing.')

        # Getting command and new sample rate
        cmd = encode.get_pcm_command(file_path, smprate, new_srate)
        if new_srate is not None: duration = int(duration / smprate * new_srate)
        smprate = new_srate is not None and new_srate or smprate

        if out is None: out = os.path.basename(file_path).rsplit('.', 1)[0]

        if meta == None: meta = encode.get_metadata(file_path)
        if img == None: img = encode.get_image(file_path)

        # Setting file extension
        if not (out.lower().endswith('.frad') or out.lower().endswith('.dsin') or out.lower().endswith('.fra') or out.lower().endswith('.dsn')):
            if profile == 0:
                if len(out) <= 8 and all(ord(c) < 128 for c in out): out += '.fra'
                else: out += '.frad'
            else:
                if len(out) <= 8 and all(ord(c) < 128 for c in out): out += '.dsn'
                else: out += '.dsin'

        if os.path.exists(out) and 'y' not in input(f'{out} Already exists. Proceed? ').lower(): sys.exit('Aborted.')

        # Fourier Transform
        try:
            start_time = time.time()
            total_bytes, total_samples = 0, 0
            cli_width = 40

            # Open FFmpeg
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            if process.stdout == None: raise FileNotFoundError('Broken pipe.')

            last = b''

            # Write file
            open(out, 'wb').write(headb.uilder(meta, img))
            with open(out, 'ab') as file:
                if verbose: print('\n\n')
                while True:
                    # bits = random.choice([12, 16, 24, 32, 48, 64]) # Random bit depth test
                    # samples_per_frame = random.choice(list(range(32, 8193))) # Random spf test
                    # profile = random.choice(list(range(2))) # Random profile test
                    # loss_level = random.choice(list(range(21))) # Random lossy level test
                    # apply_ecc = random.choice([True, False]) # Random ECC test
                    # ecc_dsize, ecc_codesize = random.choice(list(range(64, 129))), random.choice(list(range(16, 64))) # Random ECC test

                    rlen = samples_per_frame * 8 * channels
                    spf = samples_per_frame
                    while rlen < len(last):
                        spf += 128
                        rlen = spf * 8 * channels
                    if profile == 1 and len(last) != 0:
                        rlen -= len(last)

                    data = process.stdout.read(rlen) # Reading PCM
                    if not data: break               # if no data, Break

                    if len(last) != 0:
                        data = last + data
                    if profile == 1:
                        last = data[-samples_per_frame//16*8*channels:]
                    else: last = b''

                    # RAW PCM to Numpy
                    frame = np.frombuffer(data, dtype='>d').reshape(-1, channels) * gain
                    flen = len(frame)

                    # DCT
                    frame, bit_depth_frame, channels_frame, bits_efb = \
                        fourier.analogue(frame, bits, channels, little_endian, profile=profile, smprate=smprate, level=loss_level)

                    if apply_ecc: frame = ecc.encode(frame, ecc_dsize, ecc_codesize)

                    efb = headb.encode_efb(profile, apply_ecc, little_endian, bits_efb)

                    data = bytes(
                        #-- 0x00 ~ 0x0f --#
                            # Frame Signature
                            b'\xff\xd0\xd2\x97' +

                            # Frame length(Processed)
                            struct.pack('>I', len(frame)) +

                            efb + # ECC-Float Byte
                            struct.pack('>B', channels_frame - 1) +              # Channels
                            struct.pack('>B', apply_ecc and ecc_dsize or 0) +    # ECC DSize
                            struct.pack('>B', apply_ecc and ecc_codesize or 0) + # ECC Code Size

                            # Sample Rate
                            struct.pack('>I', smprate) +

                        #-- 0x10 ~ 0x1f --#
                            b'\x00'*8 +

                            # Samples in a frame per channel
                            struct.pack('>I', flen) +

                            # ISO 3309 CRC32
                            struct.pack('>I', zlib.crc32(frame)) +

                        #-- Data --#
                        frame
                    )

                    # WRITE
                    file.write(data)

                    if verbose:
                        sample_size = bit_depth_frame // 8 * channels
                        total_bytes += flen * sample_size
                        total_samples += flen
                        if profile == 1:
                            total_bytes -= flen//16 * sample_size
                            total_samples -= flen//16
                        elapsed_time = time.time() - start_time
                        bps = total_bytes / elapsed_time
                        mult = bps / smprate / sample_size
                        percent = total_samples / duration * 100
                        prgbar = int(percent / 100 * cli_width)
                        eta = (elapsed_time / (percent / 100)) - elapsed_time if percent != 0 else 'infinity'
                        print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                        print(f'Encode Speed: {(bps / 10**6):.3f} MB/s, X{mult:.3f}')
                        print(f'elapsed: {methods.tformat(elapsed_time)}, ETA {methods.tformat(eta)}')
                        print(f"[{'█'*prgbar}{' '*(cli_width-prgbar)}] {percent:.3f}% completed")

                if verbose: print('\x1b[1A\x1b[2K', end='')
        except KeyboardInterrupt:
            print('Aborting...')
            sys.exit(0)
        except Exception as e:
            if verbose: print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
            sys.exit(traceback.format_exc())
