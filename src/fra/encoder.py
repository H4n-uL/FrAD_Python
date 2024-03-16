from .common import variables, methods
from .fourier import fourier
import json, os, random, struct, subprocess, sys, time, traceback, zlib
import numpy as np
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
                duration = stream['duration_ts'] * int(stream['sample_rate']) // int(stream['time_base'][2:])
                return int(stream['channels']), int(stream['sample_rate']), stream['codec_name'], duration
        return None

    def get_pcm_command(file_path: str, osr: int, nsr: int):
        command = [
            variables.ffmpeg,
            '-v', 'quiet',
            '-i', file_path,
            '-f', 'f64le',
            '-acodec', 'pcm_f64le',
            '-vn'
        ]
        if nsr not in [osr, None]:
            command.extend(['-ar', str(nsr)])
        command.append('pipe:1')
        return command

    def get_metadata(file_path: str):
        excluded = ['major_brand', 'minor_version', 'compatible_brands', 'encoder']
        command = [
            variables.ffmpeg, '-v', 'quiet',
            '-i', file_path,
            '-f', 'ffmetadata',
            variables.meta
        ]
        subprocess.run(command)
        with open(variables.meta, 'r', encoding='utf-8') as m:
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

    def enc(file_path: str, bits: int, little_endian: bool = False,
                out: str = None, lossy: bool = False, loss_level: int = 0,
                samples_per_frame: int = 2048, gain: list = None,
                apply_ecc: bool = False, ecc_sizes: list = ['128', '20'],
                nsr: int = None,
                meta = None, img: bytes = None,
                verbose: bool = False):
        ecc_dsize = int(ecc_sizes[0])
        ecc_codesize = int(ecc_sizes[1])

        methods.cantreencode(open(file_path, 'rb').read(4))
        gain = methods.get_gain(gain)

        if not 20 >= loss_level >= 0: raise ValueError(f'Lossy compression level should be between 0 and 20.')
        if lossy and 'y' not in input('\033[1m!!!Warning!!!\033[0m\nFourier Analogue-in-Digital is designed to be an uncompressed archival codec. Compression increases the difficulty of decoding and makes data very fragile, making any minor damage likely to destroy the entire frame. Proceed? (Y/N) ').lower(): sys.exit('Aborted.')

        # Getting Audio info w. ffmpeg & ffprobe
        channels, sample_rate, codec, duration = encode.get_info(file_path)
        segmax = ((2**31-1) // (((ecc_dsize+ecc_codesize)/ecc_dsize if apply_ecc else 1) * channels * 16)//16)*2
        if samples_per_frame > segmax: raise ValueError(f'Sample size cannot exceed {segmax}.')
        if samples_per_frame < 2: raise ValueError(f'Sample size must be at least 2.')
        if samples_per_frame % 2 != 0: raise ValueError('Sample size must be multiple of 2.')

        # Getting command and new sample rate
        cmd = encode.get_pcm_command(file_path, sample_rate, nsr)
        if nsr is not None: duration = int(duration / sample_rate * nsr)
        sample_rate = nsr is not None and nsr or sample_rate

        if out is None: out = os.path.basename(file_path).rsplit('.', 1)[0]

        if meta == None: meta = encode.get_metadata(file_path)
        if img == None: img = encode.get_image(file_path)

        # Setting file extension
        if not (out.lower().endswith('.frad') or out.lower().endswith('.dsin') or out.lower().endswith('.fra') or out.lower().endswith('.dsn')):
            if len(out) <= 8 and all(ord(c) < 128 for c in out): out += '.fra'
            else: out += '.frad'
        
        if os.path.exists(out) and 'y' not in input(f'{out} Already exists. Proceed? ').lower(): sys.exit('Aborted.')

        # Fourier Transform
        try:
            start_time = time.time()
            total_bytes = 0
            cli_width = 40
            sample_size = bits // 4 * channels

            # Open FFmpeg
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

            last = b''

            # Write file
            open(out, 'wb').write(headb.uilder(meta, img))
            with open(out, 'ab') as file:
                if verbose: print('\n\n')
                while True:
                    # samples_per_frame = random.choice([i for i in range(1024, 4097, 2)]) # Random spf test
                    rlen = samples_per_frame * 8 * channels
                    if lossy and len(last) != 0:
                        rlen -= len(last)

                    data = process.stdout.read(rlen) # Reading PCM
                    if not data: break               # if no data, Break

                    if lossy:
                        data = last + data
                        last = data[-samples_per_frame//16*8*channels:]

                    # RAW PCM to Numpy
                    frame = np.frombuffer(data, dtype='<d').reshape(-1, channels) * gain

                    # MDCT
                    segment, bt = fourier.analogue(frame, bits, channels, little_endian, lossy=lossy, sample_rate=sample_rate, level=loss_level)
                    if lossy: segment = zlib.compress(segment, level=9)

                    # Applying ECC (This will make encoding hundreds of times slower)
                    # ecc_dsize, ecc_codesize = random.choice([i for i in range(64, 129)]), random.choice([i for i in range(16, 64)]) # Random ECC test
                    if apply_ecc: segment = ecc.encode(segment, ecc_dsize, ecc_codesize)

                    data = bytes(
                        #-- 0x00 ~ 0x0f --#
                            # Frame Signature
                            b'\xff\xd0\xd2\x97' +

                            # Segment length(Processed)
                            struct.pack('>I', len(segment)) +

                            headb.encode_efb(lossy, apply_ecc, little_endian, bt) + # EFB
                            struct.pack('>B', channels - 1) +                       # Channels
                            struct.pack('>B', ecc_dsize if apply_ecc else 0) +      # ECC DSize
                            struct.pack('>B', ecc_codesize if apply_ecc else 0) +   # ECC code size

                            struct.pack('>I', sample_rate) +                      # Sample Rate

                        #-- 0x10 ~ 0x1f --#
                            b'\x00'*12 +

                            # ISO 3309 CRC32
                            struct.pack('>I', zlib.crc32(segment)) +

                        #-- Data --#
                        segment
                    )

                    # WRITE
                    file.write(data)

                    if verbose:
                        total_bytes += len(frame) * sample_size
                        if lossy: total_bytes -= len(frame)//16 * sample_size
                        elapsed_time = time.time() - start_time
                        bps = total_bytes / elapsed_time
                        mult = bps / sample_rate / sample_size
                        percent = total_bytes / duration / bits * 200
                        prgbar = int(percent / 100 * cli_width)
                        eta = (elapsed_time / (percent / 100)) - elapsed_time if percent != 0 else 'infinity'
                        print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                        print(f'Encode Speed: {(bps / 10**6):.3f} MB/s, X{mult:.3f}')
                        print(f'elapsed: {elapsed_time:.3f} s, ETA {eta:.3f} s')
                        print(f"[{'█'*prgbar}{' '*(cli_width-prgbar)}] {percent:.3f}% completed")

                if verbose: print('\x1b[1A\x1b[2K', end='')
        except KeyboardInterrupt:
            print('Aborting...')
        except Exception as e:
            print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
            sys.exit(traceback.format_exc())
        sys.exit(0)
