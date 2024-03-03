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

    def get_pcm(file_path: str, osr: int, nsr: int):
        command = [
            variables.ffmpeg,
            '-v', 'quiet',
            '-i', file_path,
            '-f', 's32le',
            '-acodec', 'pcm_s32le',
            '-vn'
        ]
        if nsr not in [osr, None]:
            command.extend(['-ar', str(nsr)])
        command.append(variables.temp_pcm)
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

    def enc(file_path: str, bits: int, endian: bool = False,
                out: str = None, lossy: bool = False, loss_level: int = 0,
                samples_per_frame: int = 2048,
                apply_ecc: bool = False, ecc_sizes: list = ['128', '20'],
                nsr: int = None,
                meta = None, img: bytes = None,
                verbose: bool = False):
        ecc_dsize = int(ecc_sizes[0])
        ecc_codesize = int(ecc_sizes[1])

        if not 20 >= loss_level >= 0: raise ValueError(f'Lossy compression level should be between 0 and 20.')
        if lossy and input('\033[1m!!!Warning!!!\033[0m\nFourier Analogue-in-Digital is designed to be an uncompressed archival codec. Compression increases the difficulty of decoding and makes data very fragile, making any minor damage likely to destroy the entire frame. Proceed? (Y/N) ').lower()!='y': sys.exit('Aborted.')

        # Getting Audio info w. ffmpeg & ffprobe
        channels, sample_rate, codec = encode.get_info(file_path)
        segmax = ((2**31-1) // (((ecc_dsize+ecc_codesize)/ecc_dsize if apply_ecc else 1) * channels * 16)//2)*2
        if samples_per_frame > segmax: raise ValueError(f'Sample size cannot exceed {segmax}.')
        if samples_per_frame < 2: raise ValueError(f'Sample size must be at least 2.')
        if samples_per_frame % 2 != 0: raise ValueError('Sample size must be multiple of 2.')

        encode.get_pcm(file_path, sample_rate, nsr)
        sample_rate = nsr is not None and nsr or sample_rate

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
            brk=0

            with open(variables.temp_pcm, 'rb') as pcm, open(out, 'ab') as file:
                if verbose: print('\n\n')
                while True:
                    p = pcm.read(samples_per_frame * 4 * channels)                           # Reading PCM
                    if lossy:
                        pcm.seek(samples_per_frame//16 * -4 * channels, 1)
                        # if at the end, Break
                        if pcm.tell()%(samples_per_frame-samples_per_frame//16)!=0 or brk==1: brk += 1
                    if not p: break                                                          # if no data, Break
                    frame = np.frombuffer(p, dtype=np.int32).reshape(-1, channels)           # RAW PCM to Numpy
                    frame = frame.astype(float) / np.iinfo(np.int32).max

                    # MDCT
                    segment, bt = fourier.analogue(frame, bits, channels, endian, lossy, sample_rate, loss_level)
                    if lossy: segment = zlib.compress(segment, level=9)

                    # Applying ECC (This will make encoding hundreds of times slower)
                    if apply_ecc: segment = ecc.encode(segment, ecc_dsize, ecc_codesize)

                    data = bytes(
                        #-- 0x00 ~ 0x0f --#
                            # Frame Signature
                            b'\xff\xd0\xd2\x97' +

                            # Segment length(Processed)
                            struct.pack('>I', len(segment)) +

                            headb.encode_efb(lossy, apply_ecc, endian, bt) +      # EFB
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
                        total_bytes += len(frame) * sample_size
                        if lossy: total_bytes -= len(frame)//16 * sample_size
                        elapsed_time = time.time() - start_time
                        bps = total_bytes / elapsed_time
                        mult = bps / sample_rate / sample_size
                        percent = total_bytes / dlen / bits * 1600
                        prgbar = int(percent / 100 * cli_width)
                        eta = (elapsed_time / (percent / 100)) - elapsed_time if percent != 0 else 'infinity'
                        print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                        print(f'Encode Speed: {(bps / 10**6):.3f} MB/s, X{mult:.3f}')
                        print(f'elapsed: {elapsed_time:.3f} s, ETA {eta:.3f} s')
                        print(f"[{'█'*prgbar}{' '*(cli_width-prgbar)}] {percent:.3f}% completed")
                    if brk > 1: break

                if verbose: print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
        except KeyboardInterrupt:
            print('Aborting...')
        finally:
            os.remove(variables.temp_pcm)
            sys.exit(0)
