from .common import variables, methods
from .fourier import fourier
import numpy as np
import os, platform, shutil, struct, subprocess, sys, time, zlib
import sounddevice as sd
from .tools.ecc import ecc
from .tools.headb import headb
from .tools.dsd import dsd

class decode:
    def internal(file_path, play: bool = False, speed: float = 1, e: bool = False, verbose: bool = False):
        with open(file_path, 'rb') as f:
            # Fixed Header
            header = f.read(64)

            # File signature verification
            methods.signature(header[0x0:0x4])

            # Taking Stream info
            channels = None
            sample_rate = None
            header_length = struct.unpack('>Q', header[0x8:0x10])[0] # 0x08-8B: Total header size

            f.seek(header_length)

            # Inverse Fourier Transform #
            i = 0
            frameNo = 0

            # Getting secure framed source length
            dlen = framescount = 0
            ecc_dsize = ecc_codesize = 0
            duration = 0
            warned = False
            error_dir = []
            while True:
                frame = f.read(32)
                if not frame: break
                blocklength = struct.unpack('>I', frame[0x4:0x8])[0]        # 0x04-4B: Audio Stream Frame length
                efb = struct.unpack('>B', frame[0x8:0x9])[0]                # 0x08:    Cosine-Float Bit
                is_ecc_on, endian, float_bits = headb.decode_efb(efb)
                channels_frame = struct.unpack('>B', frame[0x9:0xa])[0] + 1 # 0x09:    Channels
                ecc_dsize = struct.unpack('>B', frame[0xa:0xb])[0]          # 0x0a:    ECC Data block size
                ecc_codesize = struct.unpack('>B', frame[0xb:0xc])[0]       # 0x0b:    ECC Code size
                srate_frame = struct.unpack('>I', frame[0xc:0x10])[0]       # 0x0c-4B: Sample rate
                crc32 = frame[0x1c:0x20]                                    # 0x1c-4B: ISO 3309 CRC32 of Audio Data
                block = f.read(blocklength)
                if e and zlib.crc32(block) != struct.unpack('>I', crc32)[0]:
                    error_dir.append(str(framescount))
                    if not warned:
                        warned = True
                        print('This file may had been corrupted. Please repack your file via \'ecc\' option for the best music experience.')

                if is_ecc_on: block = ecc.unecc(block, ecc_dsize, ecc_codesize)
                # block = zlib.decompress(block)
                ssize_dict = {0b110: 16*channels_frame, 0b101: 8*channels_frame, 0b100: 6*channels_frame, 0b011: 4*channels_frame, 0b010: 3*channels_frame, 0b001: 2*channels_frame}
                duration += (len(block) // ssize_dict[float_bits]) / (srate_frame * speed)

                dlen += len(block)
                framescount += 1
            if error_dir != []: print(f'Corrupt frames: {", ".join(error_dir)}')

            f.seek(header_length)

            try:
                # Starting stream
                if play:
                    stream = sd.OutputStream(samplerate=44100, channels=2)
                    stream.start()
                    print()
                else:
                    stream = open(variables.temp_pcm, 'ab')
                    dlen = os.path.getsize(file_path) - header_length
                    cli_width = 40
                    start_time = time.time()
                    if verbose: print('\n\n')

                while True:
                    # Reading Frame Header
                    frame = f.read(32)
                    if not frame: break
                    blocklength = struct.unpack('>I', frame[0x4:0x8])[0]        # 0x04-4B: Audio Stream Frame length
                    efb = struct.unpack('>B', frame[0x8:0x9])[0]                # 0x08:    Cosine-Float Bit
                    is_ecc_on, endian, float_bits = headb.decode_efb(efb)
                    channels_frame = struct.unpack('>B', frame[0x9:0xa])[0] + 1 # 0x09:    Channels
                    ecc_dsize = struct.unpack('>B', frame[0xa:0xb])[0]          # 0x0a:    ECC Data block size
                    ecc_codesize = struct.unpack('>B', frame[0xb:0xc])[0]       # 0x0b:    ECC Code size
                    srate_frame = struct.unpack('>I', frame[0xc:0x10])[0]       # 0x0c-4B: Sample rate
                    crc32 = frame[0x1c:0x20]                                    # 0x1c-4B: ISO 3309 CRC32 of Audio Data
                    ssize_dict = {0b110: 16*channels_frame, 0b101: 8*channels_frame, 0b100: 6*channels_frame, 0b011: 4*channels_frame, 0b010: 3*channels_frame, 0b001: 2*channels_frame}

                    # Reading Block
                    block = f.read(blocklength)

                    if is_ecc_on:
                        if e and zlib.crc32(block) != struct.unpack('>I', crc32)[0]:
                            block = ecc.decode(block, ecc_dsize, ecc_codesize)
                        else: block = ecc.unecc(block, ecc_dsize, ecc_codesize)

                    # block = zlib.decompress(block)

                    segment = fourier.digital(block, float_bits, channels_frame, endian) # Inversing

                    if play:
                        if channels != channels_frame or sample_rate != srate_frame:
                            stream = sd.OutputStream(samplerate=int(srate_frame*speed), channels=channels_frame)
                            stream.start()
                            channels, sample_rate = channels_frame, srate_frame
                        stream.write(segment.astype(np.float32))

                        i += len(segment) / (sample_rate*speed)
                        frameNo += 1
                        print('\x1b[1A\x1b[2K', end='')
                        if verbose: 
                            print(f'{(i):.3f} s / {(duration):.3f} s (Frame #{frameNo} / {framescount} Frames)')
                        else:
                            print(f'{(i):.3f} s')
                    else:
                        if channels != channels_frame or sample_rate != srate_frame:
                            if channels != None or sample_rate != None:
                                print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                                print('Warning: Fourier Analogue-in-Digital supports variable sample rates and channels, while other codecs do not.')
                                print('The decoder has only decoded the first track. The decoding of two or more tracks with variable sample rates and channels is planned for an update.')
                                return sample_rate, channels
                            channels, sample_rate = channels_frame, srate_frame
                        stream.write(np.int32(segment*np.iinfo(np.int32).max))
                        i += blocklength + 32
                        if verbose:
                            elapsed_time = time.time() - start_time
                            bps = i / elapsed_time
                            mult = bps / (ssize_dict[float_bits] * sample_rate)
                            percent = i*100 / dlen
                            b = int(percent / 100 * cli_width)
                            eta = (elapsed_time / (percent / 100)) - elapsed_time if percent != 0 else 'infinity'
                            print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                            print(f'Decode Speed: {(bps / 10**6):.3f} MB/s, X{mult:.3f}')
                            print(f'elapsed: {elapsed_time:.3f} s, ETA {eta:.3f} s')
                            print(f"[{'█'*b}{' '*(cli_width-b)}] {percent:.3f}% completed")
                stream.close()
                if play:
                    print('\x1b[1A\x1b[2K', end='')
                elif verbose:
                    print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                return sample_rate, channels
            except KeyboardInterrupt:
                if play: stream.abort()
                else:
                    print('Aborting...')
                    os.remove(variables.temp_pcm)
                sys.exit(0)

    def split_q(s):
        if s == None: 
            return None, None
        if not s[0].isdigit():
            raise ValueError('Quality format should be [{Positive integer}{c/v/a}]')
        number = ''.join(filter(str.isdigit, s))
        strategy = ''.join(filter(str.isalpha, s))
        return number, strategy

    def setaacq(quality, channels):
        if quality == None:
            if channels == 1:
                quality = 256000
            elif channels == 2:
                quality = 320000
            else: quality = 160000 * channels
        return quality

    ffmpeg_lossless = ['wav', 'flac', 'wavpack', 'tta', 'truehd', 'alac', 'dts', 'mlp']

    def ffmpeg(sample_rate, channels, codec, f, s, out, ext, quality, strategy):
        command = [
            variables.ffmpeg, '-y',
            '-loglevel', 'error',
            '-f', 's32le',
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-i', variables.temp_pcm
        ]
        command.append('-c:a')
        if codec in ['wav', 'riff']:
            command.append(f'pcm_{f}')
        else:
            command.append(codec)

        # Lossy VS Lossless
        if codec in decode.ffmpeg_lossless:
            command.append('-sample_fmt')
            command.append(s)
        else:
            # Variable bitrate quality
            if strategy == 'v' or codec == 'libvorbis':
                if quality == None: quality = '10' if codec == 'libvorbis' else '0'
                command.append('-q:a')
                command.append(quality)

            # Constant bitrate quality
            if strategy in ['c', '', None] and codec != 'libvorbis':
                if quality == None: quality = '4096000'
                if codec == 'libopus' and int(quality) > 512000:
                    quality = '512000'
                command.append('-b:a')
                command.append(quality)

        if ext == 'ogg':
            # Muxer
            command.append('-f')
            command.append(ext)

        # File name
        command.append(f'{out}.{ext}')
        subprocess.run(command)
        os.remove(variables.temp_pcm)

    def AppleAAC_macOS(sample_rate, channels, s, out, quality, strategy):
        try:
            quality = str(quality)
            command = [
                variables.ffmpeg, '-y',
                '-loglevel', 'error',
                '-f', 's32le',
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-i', variables.temp_pcm,
                '-sample_fmt', s,
                '-f', 'flac', variables.temp_flac
            ]
            subprocess.run(command)
            os.remove(variables.temp_pcm)
        except KeyboardInterrupt:
            print('Aborting...')
            os.remove(variables.temp_pcm)
            os.remove(variables.temp_flac)
            sys.exit(0)
        try:
            if strategy in ['c', '', None]: strategy = '0'
            elif strategy == 'a': strategy = '1'
            else: raise ValueError()

            command = [
                variables.aac,
                '-f', 'adts', '-d', 'aac' if int(quality) > 64000 else 'aach',
                variables.temp_flac,
                '-b', quality,
                f'{out}.aac',
                '-s', strategy
            ]
            subprocess.run(command)
            os.remove(variables.temp_flac)
        except KeyboardInterrupt:
            print('Aborting...')
            os.remove(variables.temp_flac)
            sys.exit(0)

    def AppleAAC_Windows(sample_rate, channels, out, quality):
        try:
            command = [
                variables.aac,
                '--raw', variables.temp_pcm,
                '--raw-channels', str(channels),
                '--raw-rate', str(sample_rate),
                '--raw-format', 's32l',
                '--adts',
                '-c', quality,
                '-o', f'{out}.aac',
                '-s'
            ]
            subprocess.run(command)
            os.remove(variables.temp_pcm)
        except KeyboardInterrupt:
            print('Aborting...')
            os.remove(variables.temp_pcm)
            sys.exit(0)

    def dec(file_path, out: str = None, bits: int = 32, codec: str = None, quality: str = None, e: bool = False, nsr: int = None, verbose: bool = False):
        # Decoding
        sample_rate, channels = decode.internal(file_path, e=e, verbose=verbose)
        sample_rate = methods.resample_pcm(channels, sample_rate, nsr)

        try:
            quality, strategy = decode.split_q(quality)
            # Checking name
            if out:
                out, ext = os.path.splitext(out)
                ext = ext.lstrip('.').lower()
                if codec:
                    if ext: pass
                    else:   ext = codec
                else:
                    if ext: codec = ext
                    else:   codec = ext = 'flac'
            else:
                out = os.path.basename(file_path).rsplit('.', 1)[0]
                if codec:   ext = codec
                else:       codec = ext = 'flac'

            codec = codec.lower()

            # Checking Codec and Muxers
            if codec in ['vorbis', 'opus', 'speex']:
                codec = 'lib' + codec
                if codec in ['vorbis', 'speex']:
                    ext = 'ogg'
            if codec == 'ogg': codec = 'libvorbis'
            if codec == 'mp3': codec = 'libmp3lame'

            if bits == 32:
                f = 's32le'
                s = 's32'
            elif bits == 16:
                f = 's16le'
                s = 's16'
            elif bits == 8:
                f = s = 'u8'
            else: raise ValueError(f"Illegal value {bits} for bits: only 8, 16, and 32 bits are available for decoding.")

            if quality: int(quality.replace('k', '000'))

            if (codec == 'aac' and sample_rate <= 48000 and channels <= 2) or codec in ['appleaac', 'apple_aac']:
                if strategy in ['c', 'a']: quality = decode.setaacq(quality, channels)
                if platform.system() == 'Darwin': decode.AppleAAC_macOS(sample_rate, channels, s, out, quality, strategy)
                elif platform.system() == 'Windows': decode.AppleAAC_Windows(sample_rate, channels, out, quality)
            elif codec in ['dsd', 'dff']:
                dsd.encode(sample_rate, channels, out, ext, verbose)
            elif codec not in ['pcm', 'raw']:
                decode.ffmpeg(sample_rate, channels, codec, f, s, out, ext, quality, strategy)
            else:
                shutil.move(variables.temp_pcm, f'{out}.{ext}')

        except KeyboardInterrupt:
            print('Aborting...')
            os.remove(variables.temp_pcm)
            sys.exit(0)
