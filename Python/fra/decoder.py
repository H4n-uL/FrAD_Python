from .common import variables, methods
from .cosine import cosine
from .fourier import fourier
import hashlib, shutil, os, platform, struct, subprocess, sys, time
import numpy as np
import sounddevice as sd
from .tools.ecc import ecc

class decode:
    def internal(file_path, bits: int = 32, play: bool = False, speed: float = 1, e: bool = False, verbose: bool = False):
        with open(file_path, 'rb') as f:
            # Fixed Header
            header = f.read(256)

            # File signature verification
            methods.signature(header[0x0:0x3])

            # Taking Stream info
            channels = struct.unpack('<B', header[0x3:0x4])[0] + 1   # 0x03:          Channel
            sample_rate = struct.unpack('>I', header[0x4:0x8])[0]    # 0x04-4B:       Sample rate

            header_length = struct.unpack('>Q', header[0x8:0x10])[0] # 0x08-8B:       Total header size
            efb = struct.unpack('<B', header[0x10:0x11])[0]          # 0x10:          ECC-Float Bit
            is_cosine = True if (efb >> 7 & 0b1) == 0b1 else False   # 0x10@0b111:    MDCT Toggle(Enabled if 1)
            is_ecc_on = True if (efb >> 4 & 0b1) == 0b1 else False   # 0x10@0b100:    ECC Toggle(Enabled if 1)
            float_bits = efb & 0b111                                 # 0x10@0b011-3b: Stream bit depth
            checksum_header = header[0xf0:0x100]                     # 0xf0-16B:      Stream hash

            # Reading Audio stream
            if e:
                f.seek(header_length)
                # Verifying checksum
                md5 = hashlib.md5()
                while True:
                    d = f.read(variables.hash_block_size)
                    if not d: break
                    md5.update(d)
                checksum_data = md5.digest()
                if checksum_data != checksum_header:
                    if is_ecc_on == False:
                        print(f'Checksum: on header[{checksum_header}] vs on data[{checksum_data}]')
                        raise Exception(f'{file_path} has corrupted but it has no ECC option. Decoder halted.')
                    else:
                        print(f'Checksum: on header[{checksum_header}] vs on data[{checksum_data}]')
                        raise Exception(f'{file_path} has been corrupted, Please repack your file for the best music experience.')

            dlen = os.path.getsize(file_path) - header_length
            f.seek(header_length)

            if is_cosine: sample_size = {0b011: 8*channels, 0b010: 4*channels, 0b001: 2*channels}[float_bits]
            else: sample_size = {0b011: 16*channels, 0b010: 8*channels, 0b001: 4*channels}[float_bits]
            nperseg = variables.nperseg

            # Inverse Fourier Transform #
            i = 0
            if play == True: # When playing
                try:
                    stream = sd.OutputStream(samplerate=int(sample_rate*speed), channels=channels)
                    stream.start()
                    p = sample_size * sample_rate * speed
                    if is_ecc_on: # When ECC
                        nperseg = nperseg // 128 * 148
                        p = p // 127 * 148
                    while True:
                        block = f.read(nperseg*sample_size) # Reading 2048/2368 Bytes block
                        if not block: break
                        i += len(block)
                        if is_ecc_on:
                            block = ecc.unecc(block)
                        if is_cosine: segment = (cosine.digital(block, float_bits, bits, channels, len(block)//sample_size % 2 == 1) / np.iinfo(np.int32).max).astype(np.float32) # Inversing
                        else: segment = (fourier.digital(block, float_bits, bits, channels) / np.iinfo(np.int32).max).astype(np.float32) # Inversing
                        stream.write(segment)
                        if (i // nperseg * variables.nperseg != len(block)): print('\x1b[1A\x1b[2K', end='')
                        if verbose: 
                            print(f'{(i / p):.3f} s / {(dlen / p):.3f} s (Frame #{i // nperseg // sample_size} / {dlen // nperseg // sample_size} Frames)')
                        else: 
                            print(f'{(i / p):.3f} s')
                    print('\x1b[1A\x1b[2K', end='')
                    stream.close()
                    sys.exit(0)
                except KeyboardInterrupt:
                    stream.close()
                    sys.exit(0)
            else:
                try:
                    cli_width = 40
                    with open(variables.temp_pcm, 'wb') as p:
                        if is_ecc_on: # When ECC
                            nperseg = nperseg // 128 * 148
                        start_time = time.time()
                        while True:
                            block = f.read(nperseg*sample_size) # Reading 2048/2368 Bytes block
                            if not block: break
                            if is_ecc_on:
                                block = ecc.unecc(block)
                            if is_cosine: segment = cosine.digital(block, float_bits, bits, channels, len(block)//sample_size % 2 == 1) # Inversing
                            else: segment = fourier.digital(block, float_bits, bits, channels) # Inversing
                            p.write(segment)
                            if verbose:
                                i += len(block)
                                elapsed_time = time.time() - start_time
                                bps = i / elapsed_time
                                mult = bps / (sample_size * sample_rate)
                                percent = i*100 / dlen
                                if is_ecc_on: percent = percent / 128 * 148
                                b = int(percent / 100 * cli_width)
                                if (i != len(block)): print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                                print(f'Decode Speed: {(bps / 10**6):.3f} MB/s, X{mult:.3f}')
                                print(f"[{'█'*b}{' '*(cli_width-b)}] {percent:.3f}% completed")
                        if verbose: print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                    return sample_rate, channels
                except KeyboardInterrupt:
                    print('Aborting...')
                    os.remove(variables.temp_pcm)
                    sys.exit(0)

    def setaacq(quality, channels):
        if quality == None:
            if channels == 1:
                quality = 256000
            elif channels == 2:
                quality = 320000
            else: quality = 160000 * channels
        return quality

    def ffmpeg(sample_rate, channels, codec, f, s, out, ext, quality):
        command = [
            variables.ffmpeg, '-y',
            '-loglevel', 'error',
            '-f', f,
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-i', variables.temp_pcm
        ]
        command.append('-c:a')
        if codec in ['wav', 'riff']:
            command.append(f'pcm_{f}')
        else:
            command.append(codec)

        # WAV / fLaC Sample Format
        if codec in ['wav', 'flac']:
            command.append('-sample_fmt')
            command.append(s)

        # Vorbis quality
        if codec in ['libvorbis']:
            if quality == None: quality = '10'
            command.append('-q:a')
            command.append(quality)

        # AAC, MPEG, Opus bitrate
        if codec in ['aac', 'm4a', 'libmp3lame', 'libopus']:
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

    def AppleAAC_macOS(sample_rate, channels, f, s, out, quality):
        try:
            quality = str(quality)
            command = [
                variables.ffmpeg, '-y',
                '-loglevel', 'error',
                '-f', f,
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
            command = [
                variables.aac,
                '-f', 'adts', '-d', 'aac',
                variables.temp_flac,
                '-b', quality,
                f'{out}.aac',
                '-s', '0'
            ]
            subprocess.run(command)
            os.remove(variables.temp_flac)
        except KeyboardInterrupt:
            print('Aborting...')
            os.remove(variables.temp_flac)
            sys.exit(0)

    def AppleAAC_Windows(sample_rate, channels, a, out, quality):
        try:
            command = [
                variables.aac,
                '--raw', variables.temp_pcm,
                '--raw-channels', str(channels),
                '--raw-rate', str(sample_rate),
                '--raw-format', a,
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

    def dec(file_path, out: str = None, bits: int = 32, codec: str = None, quality: str = None, e: bool = False, verbose: bool = False):
        # Decoding
        sample_rate, channels = decode.internal(file_path, bits, e=e, verbose=verbose)

        try:
            # Checking name
            if out:
                out, ext = os.path.splitext(out)
                ext = ext.lstrip('.').lower()
                if codec:
                    if ext: pass
                    else:   ext = codec
                else:
                    if      ext: codec = ext
                    else:   codec = ext = 'flac'
            else:
                if codec:   out = 'restored'; ext = codec
                else:       codec = ext = 'flac'; out = 'restored'

            # Checking Codec and Muxers
            if codec == 'vorbis' or codec == 'opus':
                codec = 'lib' + codec
                ext = 'ogg'
            if codec == 'ogg': codec = 'libvorbis'
            if codec == 'mp3': codec = 'libmp3lame'

            if bits == 32:
                f = 's32le'
                a = 's32l'
                s = 's32'
            elif bits == 16:
                f = 's16le'
                a = 's16l'
                s = 's16'
            elif bits == 8:
                f = a = s = 'u8'
            else: raise ValueError(f"Illegal value {bits} for bits: only 8, 16, and 32 bits are available for decoding.")

            if quality: int(quality.replace('k', '000'))

            if (codec == 'aac' and sample_rate <= 48000 and channels <= 2) or codec in ['appleaac', 'apple_aac']:
                quality = decode.setaacq(quality, channels)
                if platform.system() == 'Darwin': decode.AppleAAC_macOS(sample_rate, channels, f, s, out, quality)
                elif platform.system() == 'Windows': decode.AppleAAC_Windows(sample_rate, channels, a, out, quality)
            elif codec not in ['pcm', 'raw']:
                decode.ffmpeg(sample_rate, channels, codec, f, s, out, ext, quality)
            else:
                shutil.move(variables.temp_pcm, f'{out}.{ext}')

        except KeyboardInterrupt:
            print('Aborting...')
            os.remove(variables.temp_pcm)
            sys.exit(0)