from .ffpath import ff
from .fourier import fourier
import hashlib
import os
import struct
import subprocess
from .tools.ecc import ecc

class decode:
    def internal(file_path, bits: int = 32):
        with open(file_path, 'rb') as f:
            header = f.read(256)

            signature = header[0x0:0xa]
            if signature != b'\x7e\x8b\xab\x89\xea\xc0\x9d\xa9\x68\x80':
                raise Exception('This is not Fourier Analogue file.')

            header_length = struct.unpack('<Q', header[0xa:0x12])[0]
            sample_rate = int.from_bytes(header[0x12:0x15], 'little')
            cfb = struct.unpack('<B', header[0x15:0x16])[0]
            cb = (cfb >> 3) + 1
            fb = cfb & 0b111
            is_ecc_on = True if (struct.unpack('<B', header[0x16:0x17])[0] >> 7) == 0b1 else False
            checksum_header = header[0xf0:0x100]

            f.seek(header_length)

            data = f.read()
            checksum_data = hashlib.md5(data).digest()
            if is_ecc_on == False:
                if checksum_data == checksum_header:
                    pass
                else:
                    print(f'Checksum: on header[{checksum_header}] vs on data[{checksum_data}]')
                    raise Exception('File has corrupted but it has no ECC option. Decoder halted.')
            else:
                if checksum_data == checksum_header:
                    chunks = ecc.split_data(data, 148)
                    data =  b''.join([bytes(chunk[:128]) for chunk in chunks])
                else:
                    print(f'{file_path} has been corrupted, Please repack your file for the best music experience.')
                    print(f'Checksum: on header[{checksum_header}] vs on data[{checksum_data}]')
                    data = ecc.decode(data)

            restored = fourier.digital(data, fb, bits, cb)
            return restored, sample_rate

    def dec(file_path, out: str = None, bits: int = 32, codec: str = None, quality: str = None):
        restored, sample_rate = decode.internal(file_path, bits)

        if out is None and codec is None: codec = ext = 'flac'; out = 'restored'
        else:
            out, ext = os.path.splitext(out)
            if ext is None and codec is None: codec = ext = 'flac'
            elif ext == '': ext = codec
            elif codec is None: codec = ext = ext.lstrip('.').lower()
            else: ext = ext.lstrip('.').lower()

        channels = restored.shape[1] if len(restored.shape) > 1 else 1
        raw_audio = restored.tobytes()

        if codec == 'vorbis' or codec == 'opus':
            codec = 'lib' + codec

        if bits == 32:
            f = 's32le'
            s = 's32'
        elif bits == 16:
            f = 's16le'
            s = 's16'
        elif bits == 8:
            f = s = 'u8'
        else: raise ValueError(f"Illegal value {bits} for bits: only 8, 16, and 32 bits are available for decoding.")

        command = [
            ff.mpeg, '-y',
            '-loglevel', 'error',
            '-f', f,
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-i', 'pipe:0'
        ]
        if codec not in ['pcm', 'raw']:
            command.append('-c:a')
            if codec == 'wav':
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
            if codec in ['aac', 'm4a', 'mp3', 'libopus']:
                if quality == None: quality = '4096k'
                if codec == 'libopus' and int(quality.replace('k', '000')) > 512000:
                    quality = '512k'
                command.append('-b:a')
                command.append(quality)

            command.append('-f')
            command.append(codec)

            # File name
            command.append(f'{out}.{ext}')
            subprocess.run(command, input=raw_audio)
        else:
            with open(f'{out}.{ext}', 'wb') as f:
                f.write(raw_audio)
