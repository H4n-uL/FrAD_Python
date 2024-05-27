from .common import variables, methods
from .fourier import fourier
from .header import header
# import matplotlib.pyplot as plt
# from scipy.fft import dct
import numpy as np
import atexit, math, os, platform, shutil, struct, subprocess, sys, tempfile, time, traceback, zlib
import sounddevice as sd
from .tools.ecc import ecc
from .tools.headb import headb
from .tools.dsd import dsd

RM_CLI = '\x1b[1A\x1b[2K'

class ASFH:
    def __init__(self): pass

    def update(self, header: bytes):
        self.frmlen = struct.unpack('>I', header[0x4:0x8])[0]       # 0x04-4B: Audio Stream Frame length
        self.profile, self.ecc, self.endian, self.float_bits = headb.decode_efb(struct.unpack('>B', header[0x8:0x9])[0]) # 0x08: EFloat Byte
        self.chnl = struct.unpack('>B', header[0x9:0xa])[0] + 1     # 0x09:    Channels
        self.ecc_dsize = struct.unpack('>B', header[0xa:0xb])[0]    # 0x0a:    ECC Data block size
        self.ecc_codesize = struct.unpack('>B', header[0xb:0xc])[0] # 0x0b:    ECC Code size
        self.srate = struct.unpack('>I', header[0xc:0x10])[0]       # 0x0c-4B: Sample rate
        self.fsize = struct.unpack('>I', header[0x18:0x1c])[0]      # 0x18-4B: Samples in a frame per channel
        self.crc32 = header[0x1c:0x20]                              # 0x1c-4B: ISO 3309 CRC32 of Audio Data

filelist = []

@atexit.register
def cleanup():
    for file, _, _ in filelist:
        try:
            if os.path.exists(file): os.remove(file)
        except: pass

class decode:
    @staticmethod
    def internal(file_path: str, **kwargs):
        speed = kwargs.get('speed', 1)
        play = kwargs.get('play', False)
        ispipe = kwargs.get('pipe', False)
        fix_error = kwargs.get('ecc', False)
        gain = kwargs.get('gain', 1)
        verbose = kwargs.get('verbose', False)
        global filelist
        with open(file_path, 'rb') as f:

# ------------------------------ Header verification ----------------------------- #
# This block verifies the file signature and gets the total header size.
# ESSENTIAL

            # Fixed Header
            head = f.read(64)

            # File signature verification
            ftype = methods.signature(head[0x0:0x4])
            # Taking Stream info
            channels = int()
            smprate = int()

            # Container starts with 'fRad', and Stream starts with 0xffd0d297
            if ftype == 'container':
                head_len = struct.unpack('>Q', head[0x8:0x10])[0] # 0x08-8B: Total header size
            elif ftype == 'stream': head_len = 0
            f.seek(head_len)
            t_accr = dict()
            bytes_accr = frameNo = 0
            dlen = framescount = duration = 0
            asfh = ASFH()

# ----------------------------- Getting source length ---------------------------- #
# This block gets the total length of the stream and the number of frames included.
# This is for the progress bar and playback duration.
# OPTIONAL: for minimal implementation, you can skip this block but recommended.

            warned = False
            error_dir = []
            fhead = None
            while True:
                # Finding Audio Stream Frame Header(AFSH)
                if fhead is None: fhead = f.read(4)
                if fhead != b'\xff\xd0\xd2\x97':
                    hq = f.read(1)
                    if not hq:
                        if asfh.profile in [1, 2]: duration += asfh.fsize // 16 / asfh.srate
                        break
                    fhead = fhead[1:]+hq
                    continue
                asfh.update(fhead+f.read(28))
                data = f.read(asfh.frmlen)
                if fix_error and zlib.crc32(data) != struct.unpack('>I', asfh.crc32)[0]:
                    error_dir.append(str(framescount))
                    if not warned: warned = True; print("This file may had been corrupted. Please repack your file via 'ecc' option for the best music experience.")

                duration += asfh.fsize / asfh.srate
                if asfh.profile in [1, 2]: duration -= asfh.fsize//16 / asfh.srate

                dlen += asfh.frmlen
                framescount += 1
                fhead = None

            # show error frames
            if error_dir != []: print(f'Corrupt frames: {", ".join(error_dir)}')
            duration /= speed
            f.seek(head_len)

# ----------------------------------- Metadata ----------------------------------- #
# This block parses and shows the metadata and image data from the header.
# OPTIONAL: i don't even activate this block cuz it makes cli becomes messy

            # if verbose: 
            #     meta, img = header.parse(file_path)
            #     if meta:
            #         meta_tlen = max([len(m[0]) for m in meta])
            #         print('Metadata')
            #         for m in meta:
            #             if '\n' in m[1]:
            #                 m[1] = m[1].replace('\n', f'\n{" "*max(meta_tlen+2, 19)}: ')
            #             print(f'  {m[0].ljust(17, ' ')}: {m[1]}')

# ----------------------------------- Decoding ----------------------------------- #
# This block decodes FrAD stream to PCM stream and writes it on stdout or a file.
# ESSENTIAL

            stdoutstrm = sd.OutputStream()
            tempfstrm = open(os.devnull, 'wb')
            try:
                # Starting stream
                printed = False
                bps, bpstot = 0, 0
                dlen = os.path.getsize(file_path) - head_len
                cli_width = 40
                start_time = time.time()
                fhead, prev, frame = None, None, np.array(0)

    # ----------------------------- Main decode loop ----------------------------- #
                while True:
                    # Finding Audio Stream Frame Header(AFSH)
                    if fhead is None: fhead = f.read(4)
                    if fhead != b'\xff\xd0\xd2\x97':
                        hq = f.read(1)
                        if not hq:
                            if prev is not None:
                                if play: stdoutstrm.write(frame.astype(np.float32))
                                else:    tempfstrm.write(frame.astype('>f8').tobytes())
                            break
                        fhead = fhead[1:]+hq
                        continue

                    # Parsing ASFH
                    asfh.update(fhead+f.read(28))
                    # Reading Block
                    data: bytes = f.read(asfh.frmlen)

                    # Decoding ECC
                    if asfh.ecc:
                        if fix_error and zlib.crc32(data) != struct.unpack('>I', asfh.crc32)[0]:
                            data = ecc.decode(data, asfh.ecc_dsize, asfh.ecc_codesize)
                        else: data = ecc.unecc(data, asfh.ecc_dsize, asfh.ecc_codesize)

                    # Decoding
                    frame: np.ndarray = fourier.digital(data, asfh.float_bits, asfh.chnl, asfh.endian, profile=asfh.profile, smprate=asfh.srate, fsize=asfh.fsize) * gain

                    # 1/16 Overlapping
                    if prev is not None:
                        fade_in = np.linspace(0, 1, len(prev))
                        fade_out = np.linspace(1, 0, len(prev))
                        for c in range(asfh.chnl):
                            frame[:len(prev), c] = \
                            (frame[:len(prev), c] * fade_in) +\
                            (prev[:, c]           * fade_out)
                    if asfh.profile in [1, 2]:
                        prev = frame[-len(frame)//16:]
                        frame = frame[:-len(prev)]
                    else: prev = None

                    # if channels and sample rate changed
                    if channels != asfh.chnl or smprate != asfh.srate:
                        channels, smprate = asfh.chnl, asfh.srate
                        if play: # recreate stream
                            stdoutstrm = sd.OutputStream(samplerate=int(asfh.srate*speed), channels=asfh.chnl)
                            stdoutstrm.start()
                        else: # add a new file
                            tempfstrm.close()
                            tempfstrm = open(tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.pcm').name, 'wb')
                            filelist.append([tempfstrm.name, channels, smprate])

                    # Write PCM Stream
                    if play:
                        # Play block
                        stdoutstrm.write(frame.astype(np.float32))
                    else:
                        # Write block
                        if ispipe: sys.stdout.buffer.write(frame.astype('>f8').tobytes())
                        else: tempfstrm.write(frame.astype('>f8').tobytes())

                    # Verbose block

                    frameNo += 1
                    try: t_accr[smprate*speed] += len(frame)
                    except: t_accr[smprate*speed] = len(frame)
                    t_sec = sum([t_accr[k] / k for k in t_accr])
                    bytes_accr += asfh.frmlen + 32
                    if play:
                        # for i in range(asfh.chnl):
                        #     plt.subplot(asfh.chnl, 1, i+1)
                        #     # plt.plot(frame[:, i], alpha=0.5)
                        #     y = np.abs(dct(frame[:, i]) / len(frame))
                        #     plt.fill_between(range(1, len(y)+1), y, -y, edgecolor='none')
                        #     plt.xscale('log', base=2)
                        #     plt.ylim(-1, 1)
                        # plt.draw()
                        # plt.pause(0.000001)
                        # plt.clf()

                        bps = (((asfh.frmlen+len(fhead)) * 8) * asfh.srate / len(frame))
                        bpstot += bps
                        depth = [[12,16,24,32,48,64,128],[8,12,16,24,32,48,64],[8,12,16,24,32,48,64]][asfh.profile][asfh.float_bits]
                        lgs = int(math.log(asfh.srate, 1000))
                        lgv = int(math.log(bpstot/frameNo, 1000))
                        if verbose:
                            if printed: print(RM_CLI*5, end='')
                            print(f'{methods.tformat(t_sec)} / {methods.tformat(duration)} (Frame #{frameNo} / {framescount} Frame{(framescount!=1)*"s"})')
                            print(f'{depth}b@{asfh.srate/10**(lgs*3)} {['','k','M','G','T'][lgs]}Hz {not asfh.endian and"B"or"L"}E {asfh.chnl} channel{(asfh.chnl!=1)*"s"}')
                            lgf = int(math.log(bps, 1000))
                            print(f'Profile {asfh.profile}, ECC{asfh.ecc and f": {asfh.ecc_dsize}/{asfh.ecc_codesize}" or " disabled"}')
                            print(f'{len(frame)} sample{len(frame)!=1 and"s"or""}, {asfh.frmlen} Byte{(asfh.frmlen!=1)*"s"} per frame')
                            print(f'{bps/10**(lgf*3):.3f} {['','k','M','G','T'][lgf]}bps per-frame, {bpstot/frameNo/10**(lgv*3):.3f} {['','k','M','G','T'][lgv]}bps average')
                        else:
                            if printed: print(RM_CLI, end='')
                            cq = {1:'Mono',2:'Stereo',4:'Quad',6:'5.1 Surround',8:'7.1 Surround'}.get(asfh.chnl, f'{asfh.chnl} ch')
                            print(f'{methods.tformat(t_sec)} / {methods.tformat(duration)}, {asfh.profile==0 and f"{depth}b@"or f"{bpstot/frameNo/10**(lgv*3):.3f} {['','k','M','G','T'][lgv]}bps "}{asfh.srate/10**(lgs*3)} {['','k','M','G','T'][lgs]}Hz {cq}')
                        printed = True

                    else:
                        if verbose and not ispipe:
                            elapsed_time = time.time() - start_time
                            bps = bytes_accr / elapsed_time
                            lgb = int(math.log(bps, 1000))
                            mult = t_accr / (time.time() - start_time)
                            percent = bytes_accr*100 / dlen
                            b = int(percent / 100 * cli_width)
                            eta = (elapsed_time / (percent / 100)) - elapsed_time if percent != 0 else 'infinity'
                            if printed: print(RM_CLI*3, end='')
                            print(f'Decode Speed: {(bps/10**(lgb*3)):.3f} {['','k','M','G','T'][lgb]}B/s, X{mult:.3f}')
                            print(f'elapsed: {methods.tformat(elapsed_time)}, ETA {methods.tformat(eta)}')
                            print(f"[{'█'*b}{' '*(cli_width-b)}] {percent:.3f}% completed")
                            printed = True
                    fhead = None

                if printed and (play or verbose):
                    print(RM_CLI, end='')
                    if play and verbose: print(RM_CLI*4, end='')
                stdoutstrm.stop()
                tempfstrm.close()
            except KeyboardInterrupt:
                stdoutstrm.abort()
                stdoutstrm.close()
                if not play:
                    print('Aborting...')
                sys.exit(0)

    @staticmethod
    def split_q(s) -> tuple[int|None, str]:
        if s == None:
            return None, 'c'
        if not s[0].isdigit():
            print('Quality format should be [{Positive integer}{c/v/a}]')
            sys.exit(1)
        number = int(''.join(filter(str.isdigit, s)))
        strategy = ''.join(filter(str.isalpha, s))
        return number, strategy

    @staticmethod
    def setaacq(quality: int | None, channels: int):
        if quality == None:
            if channels == 1:
                return 256000
            elif channels == 2:
                return 320000
            else: return 160000 * channels
        return quality

    ffmpeg_lossless = ['wav', 'flac', 'wavpack', 'tta', 'truehd', 'alac', 'dts', 'mlp']

    @staticmethod
    def directcmd(temp_pcm, smprate, channels, ffmpeg_cmd):
        command = [
            variables.ffmpeg, '-y',
            '-loglevel', 'error',
            '-f', 'f64be',
            '-ar', str(smprate),
            '-ac', str(channels),
            '-i', temp_pcm,
            '-i', variables.meta,
        ]
        command.extend(['-map_metadata', '1', '-map', '0:a'])
        command.extend(ffmpeg_cmd)
        subprocess.run(command)

    @staticmethod
    def ffmpeg(temp_pcm, smprate, channels, codec, f, s, out, ext, quality, strategy, new_srate):
        command = [
            variables.ffmpeg, '-y',
            '-loglevel', 'error',
            '-f', 'f64be',
            '-ar', str(smprate),
            '-ac', str(channels),
            '-i', temp_pcm,
            '-i', variables.meta,
        ]
        if os.path.exists(f'{variables.meta}.image'):
            command.extend(['-i', f'{variables.meta}.image', '-c:v', 'copy'])

        command.extend(['-map_metadata', '1', '-map', '0:a'])

        if os.path.exists(f'{variables.meta}.image'):
            command.extend(['-map', '2:v'])

        if new_srate is not None and new_srate != smprate: command.extend(['-ar', str(new_srate)])

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

    @staticmethod
    def AppleAAC_macOS(temp_pcm, smprate, channels, out, quality, strategy):
        try:
            quality = str(quality)
            command = [
                variables.ffmpeg, '-y',
                '-loglevel', 'error',
                '-f', 'f64be',
                '-ar', str(smprate),
                '-ac', str(channels),
                '-i', temp_pcm,
                '-sample_fmt', 's32',
                '-f', 'flac', variables.temp_flac
            ]
            subprocess.run(command)
        except KeyboardInterrupt:
            print('Aborting...')
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
        except KeyboardInterrupt:
            print('Aborting...')
            sys.exit(0)

    @staticmethod
    def AppleAAC_Windows(temp_pcm, smprate, channels, out, quality, new_srate):
        try:
            command = [
                variables.aac,
                '--raw', temp_pcm,
                '--raw-channels', str(channels),
                '--raw-rate', str(smprate),
                '--raw-format', 'f64l',
                '--adts',
                '-c', str(quality),
            ]
            if new_srate is not None and new_srate != smprate: command.extend(['--rate', str(new_srate)])
            command.extend([
                '-o', f'{out}.aac',
                '-s'
            ])
            subprocess.run(command)
        except KeyboardInterrupt:
            print('Aborting...')
            sys.exit(0)

    @staticmethod
    def dec(file_path, **kwargs):
        # FFmpeg Command
        ffmpeg_cmd = kwargs.get('directcmd', None)

        # Output file
        out: str = kwargs.get('out', None)

        # Output file specifications
        bits: int = kwargs.get('bits', 32)
        codec: str = kwargs.get('codec', None)
        quality: str = kwargs.get('quality', None)

        # Error correction
        ecc: bool = kwargs.get('ecc', False)

        # Audio settings
        gain: float = kwargs.get('gain', 1)
        new_srate: int = kwargs.get('srate', None)

        # CLI
        verbose: bool = kwargs.get('verbose', False)

        # Decoding
        decode.internal(file_path, ecc=ecc, gain=gain, pipe=(out=='pipe'and True or False), verbose=verbose)
        if out == 'pipe': sys.exit(0)
        header.parse_to_ffmeta(file_path, variables.meta)

        try:
            if quality: quality = quality.replace('k', '000').replace('M', '000000').replace('G', '000000000').replace('T', '000000000000')
            q, strategy = decode.split_q(quality)
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
                if codec in ['vorbis', 'speex']:
                    ext = 'ogg'
                codec = 'lib' + codec
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
            else: raise ValueError(f'Illegal value {bits} for bits: only 8, 16, and 32 bits are available for decoding.')

            for z in range(len(filelist)):
                temp_pcm, channels, smprate = filelist[z]

                if ffmpeg_cmd is not None:
                    decode.directcmd(temp_pcm, smprate, channels, ffmpeg_cmd)
                else:
                    if (codec == 'aac' and smprate <= 48000 and channels <= 2) or codec in ['appleaac', 'apple_aac']:
                        if strategy in ['c', 'a']: q = decode.setaacq(q, channels)
                        if platform.system() == 'Darwin': decode.AppleAAC_macOS(temp_pcm, smprate, channels, (z==0 and out or f'{out}.{z}'), q, strategy)
                        elif platform.system() == 'Windows': decode.AppleAAC_Windows(temp_pcm, smprate, channels, (z==0 and out or f'{out}.{z}'), q, new_srate)
                    elif codec in ['dsd', 'dff']:
                        dsd.encode(temp_pcm, smprate, channels, (z==0 and out or f'{out}.{z}'), ext, verbose)
                    elif codec not in ['pcm', 'raw']:
                        decode.ffmpeg(temp_pcm, smprate, channels, codec, f, s, (z==0 and out or f'{out}.{z}'), ext, q, strategy, new_srate)
                    else:
                        shutil.move(temp_pcm, (z==0 and f'{out}.{ext}' or f'{out}.{z}.{ext}'))
                os.remove(temp_pcm)

        except KeyboardInterrupt: print('Aborting...')
        except Exception as exc: sys.exit(traceback.format_exc())
        finally: sys.exit(0)
