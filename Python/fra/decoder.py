from .common import variables, methods
from .fourier import fourier
import hashlib, shutil, os, platform, struct, subprocess
import numpy as np
import sounddevice as sd
from .tools.ecc import ecc

class decode:
    def internal(file_path, bits: int = 32, play: bool = False, speed: float = 1, e: bool = False):
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
            is_ecc_on = True if (efb >> 4 & 0b1) == 0b1 else False   # 0x10@0b100:    ECC Toggle(Enabled if 1)
            float_bits = efb & 0b111                                 # 0x10@0b011-3b: Stream bit depth
            checksum_header = header[0xf0:0x100]                     # 0xf0-16B:      Stream hash

            # Reading Audio stream
            if e:
                f.seek(header_length)
                # Verifying checksum
                checksum_data = hashlib.md5(f.read()).digest()
                if checksum_data != checksum_header:
                    if is_ecc_on == False:
                        print(f'Checksum: on header[{checksum_header}] vs on data[{checksum_data}]')
                        raise Exception(f'{file_path} has corrupted but it has no ECC option. Decoder halted.')
                    else:
                        print(f'Checksum: on header[{checksum_header}] vs on data[{checksum_data}]')
                        raise Exception(f'{file_path} has been corrupted, Please repack your file for the best music experience.')

            dlen = os.path.getsize(file_path) - header_length
            f.seek(header_length)

            sample_size = {0b011: 16*channels, 0b010: 8*channels, 0b001: 4*channels}[float_bits]
            nperseg = variables.nperseg

            p = sample_size * sample_rate * speed
            # Inverse Fourier Transform
            if play == True: # When playing
                stream = sd.OutputStream(samplerate=sample_rate*speed, channels=channels)
                stream.start()
                if is_ecc_on: # When ECC
                    nperseg = nperseg // 128 * 148
                    for i in range(0, dlen, nperseg*sample_size):
                        print(f'{(i // 148 * 128) / p:.3f} s / {(dlen // 148 * 128) / p:.3f} s (Frame #{i // nperseg // sample_size} / {dlen // nperseg // sample_size} Frames, Sample #{i * 2048 // nperseg // sample_size} / {dlen * 2048 // nperseg // sample_size} Samples)')
                        block = f.read(nperseg*sample_size) # Reading 2368 Bytes block
                        chunks = ecc.split_data(block, 148) # Carrying first 128 Bytes data from 148 Bytes chunk
                        block =  b''.join([bytes(chunk[:128]) for chunk in chunks])
                        segment = (fourier.digital(block, float_bits, bits, channels) / np.iinfo(np.int32).max).astype(np.float32) # Inversing
                        stream.write(segment)
                        print('\x1b[1A\x1b[2K', end='')
                else:         # When No ECC
                    for i in range(0, dlen, nperseg*sample_size):
                        print(f'{i / p:.3f} s / {dlen / p:.3f} s (Frame #{i // nperseg // sample_size} / {dlen // nperseg // sample_size} Frames, Sample #{i * 2048 // nperseg // sample_size} / {dlen * 2048 // nperseg // sample_size} Samples)')
                        block = f.read(nperseg*sample_size) # Reading 2048 Bytes block
                        segment = (fourier.digital(block, float_bits, bits, channels) / np.iinfo(np.int32).max).astype(np.float32) # Inversing
                        stream.write(segment)
                        print('\x1b[1A\x1b[2K', end='')
                stream.stop()
                return
            else:
                with open(variables.temp_pcm, 'wb') as p:
                    if is_ecc_on: # When ECC
                        nperseg = nperseg // 128 * 148
                        for i in range(0, dlen, nperseg*sample_size):
                            block = f.read(nperseg*sample_size) # Reading 2368 Bytes block
                            chunks = ecc.split_data(block, 148) # Carrying first 128 Bytes data from 148 Bytes chunk
                            block =  b''.join([bytes(chunk[:128]) for chunk in chunks])
                            segment = fourier.digital(block, float_bits, bits, channels) # Inversing
                            p.write(segment)
                    else:         # When No ECC
                        for i in range(0, dlen, nperseg*sample_size):
                            block = f.read(nperseg*sample_size) # Reading 2048 Bytes block
                            segment = fourier.digital(block, float_bits, bits, channels) # Inversing
                            p.write(segment)
                return sample_rate, channels

    def ffmpeg(sample_rate, channels, codec, f, s, out, ext, quality):
        if ext in ['aac', 'm4a']: print('FFmpeg doesn\'t support AAC/M4A Muxer. Switching to MP4...'); ext = 'mp4'
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
            if quality == None: quality = '4096k'
            if codec == 'libopus' and int(quality.replace('k', '000')) > 512000:
                quality = '512k'
            command.append('-b:a')
            command.append(quality)

        # Muxer
        command.append('-f')
        command.append(ext)

        # File name
        command.append(f'{out}.{ext}')
        subprocess.run(command)
        os.remove(variables.temp_pcm)

    def AppleAAC(sample_rate, channels, a, out, ext, quality):
        command = [
            variables.aac,
            '--raw-format', a,
            '--raw-rate', str(sample_rate),
            '--raw-channels', str(channels),
            '--raw', variables.temp_pcm
        ]
        if quality == None: quality = '4096'
        command.append('-c')
        command.append(quality)

        command.append('-o')
        command.append(f'{out}.{ext}')
        command.append('-s')
        subprocess.run(command)
        os.remove(variables.temp_pcm)

    def dec(file_path, out: str = None, bits: int = 32, codec: str = None, quality: str = None, e: bool = False):
        # Decoding
        sample_rate, channels = decode.internal(file_path, bits, e=e)

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

        if codec == 'aac' and sample_rate <= 48000 and platform.system() == 'Darwin':
            decode.AppleAAC(sample_rate, channels, a, out, ext, quality)
        elif codec not in ['pcm', 'raw']:
            decode.ffmpeg(sample_rate, channels, codec, f, s, out, ext, quality)
        else:
            shutil.move(variables.temp_pcm, f'{out}.{ext}')
