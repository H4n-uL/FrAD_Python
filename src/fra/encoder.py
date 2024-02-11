from .common import variables, methods
from .fourier import fourier
import json, os, struct, subprocess, sys, time, zlib
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

    def enc(file_path: str, bits: int, endian: bool = False,
                out: str = None, samples_per_block: int = 2048,
                apply_ecc: bool = False, ecc_sizes: list = ['128', '20'],
                nsr: int = None,
                meta = None, img: bytes = None,
                verbose: bool = False):
        ecc_dsize = int(ecc_sizes[0])
        ecc_codesize = int(ecc_sizes[1])

        # Getting Audio info w. ffmpeg & ffprobe
        channels, sample_rate, codec = encode.get_info(file_path)
        segmax = (2**32 // (((ecc_dsize+ecc_codesize)/ecc_dsize if apply_ecc else 1) * channels * bits // 8)//4)*4
        if samples_per_block > segmax: raise ValueError(f'Sample size cannot exceed {segmax}.')
        if samples_per_block < 4: raise ValueError(f'Sample size must be at least 4.')
        if samples_per_block % 4 != 0: raise ValueError('Sample size must be multiple of 4.')

        encode.get_pcm(file_path)
        sample_rate = methods.resample_pcm(channels, sample_rate, nsr)

        if out is None: out = os.path.basename(file_path).rsplit('.', 1)[0]

        if meta == None: meta = encode.get_metadata(file_path)
        if img == None: img = encode.get_image(file_path)

        # Setting file extension
        if not (out.lower().endswith('.frad') or out.lower().endswith('.dsin') or out.lower().endswith('.fra') or out.lower().endswith('.dsn')):
            if len(out) <= 8 and all(ord(c) < 128 for c in out): out += '.fra'
            else: out += '.frad'

        with open(out, 'wb') as file:
            h = headb.uilder(meta, img)
            file.write(h)

        # Fourier Transform
        try:
            start_time = time.time()
            total_bytes = 0
            cli_width = 40
            sample_size = bits // 4 * channels
            dlen = os.path.getsize(variables.temp_pcm)

            with open(variables.temp_pcm, 'rb') as pcm, open(out, 'ab') as file:
                if verbose: print('\n\n')
                while True:
                    p = pcm.read(samples_per_block * 4 * channels)                 # Reading PCM
                    if not p: break                                                # if no data, Break
                    block = np.frombuffer(p, dtype=np.int32).reshape(-1, channels) # RAW PCM to Numpy
                    block = block.astype(float) / np.iinfo(np.int32).max
                    segment = fourier.analogue(block, bits, channels, True)              # Fourier Transform

                    # segment = zlib.compress(segment)

                    # Applying ECC (This will make encoding thousands of times slower)
                    if apply_ecc: segment = ecc.encode(segment, ecc_dsize, ecc_codesize)

                    data = bytes(
                        #-- 0x00 ~ 0x0f --#
                            # Block Signature
                            b'\xff\xd0\xd2\x97' +

                            # Segment length(Processed)
                            struct.pack('>I', len(segment)) +

                            headb.encode_efb(apply_ecc, True, bits) +                   # EFB
                            struct.pack('>B', channels - 1) +                     # Channels
                            struct.pack('>B', ecc_dsize if apply_ecc else 0) +    # ECC DSize
                            struct.pack('>B', ecc_codesize if apply_ecc else 0) + # ECC code size

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
                        total_bytes += len(block) * sample_size
                        elapsed_time = time.time() - start_time
                        bps = total_bytes / elapsed_time
                        mult = bps / sample_rate / sample_size
                        percent = total_bytes / dlen / bits * 1600
                        b = int(percent / 100 * cli_width)
                        eta = (elapsed_time / (percent / 100)) - elapsed_time if percent != 0 else 'infinity'
                        print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                        print(f'Encode Speed: {(bps / 10**6):.3f} MB/s, X{mult:.3f}')
                        print(f'elapsed: {elapsed_time:.3f} s, ETA {eta:.3f} s')
                        print(f"[{'█'*b}{' '*(cli_width-b)}] {percent:.3f}% completed")

                if verbose: print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
        except KeyboardInterrupt:
            print('Aborting...')
        finally:
            os.remove(variables.temp_pcm)
            sys.exit(0)
