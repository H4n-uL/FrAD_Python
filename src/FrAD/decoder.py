from .common import variables, methods, terminal
from .fourier import fourier
from .header import header
import numpy as np
from .profiles.prf import profiles
import atexit, io, math, os, platform, shutil, struct,\
       subprocess, sys, tempfile, time, traceback, zlib
import sounddevice as sd
from .tools.ecc import ecc
from .tools.headb import headb

RM_CLI = '\x1b[1A\x1b[2K'

class ASFH:
    def __init__(self): pass

    def update(self, file: io.BufferedReader) -> bool:
        fhead = variables.FRM_SIGN + file.read(5)
        self.frmbytes = struct.unpack('>I', fhead[0x4:0x8])[0]        # 0x04-4B: Audio Stream Frame length
        self.profile, self.ecc, self.endian, self.float_bits = headb.decode_pfb(fhead[0x8:0x9]) # 0x08: EFloat Byte

        if self.profile in profiles.LOSSLESS:
            fhead += file.read(23)
            self.chnl = struct.unpack('>B', fhead[0x9:0xa])[0] + 1     # 0x09:    Channels
            self.ecc_dsize = struct.unpack('>B', fhead[0xa:0xb])[0]    # 0x0a:    ECC Data block size
            self.ecc_codesize = struct.unpack('>B', fhead[0xb:0xc])[0] # 0x0b:    ECC Code size
            self.srate = struct.unpack('>I', fhead[0xc:0x10])[0]       # 0x0c-4B: Sample rate
            self.fsize = struct.unpack('>I', fhead[0x18:0x1c])[0]      # 0x18-4B: Samples in a frame per channel
            self.crc = fhead[0x1c:0x20]                                # 0x1c-4B: ISO 3309 CRC32 of Audio Data

        if self.profile in profiles.COMPACT:
            fhead += file.read(3)
            self.chnl, self.srate, self.fsize, force_flush = headb.decode_css_prf1(fhead[0x9:0xb])
            if force_flush: return True
            self.overlap = struct.unpack('>B', fhead[0xb:0xc])[0]      # 0x0b: Overlap rate
            if self.overlap != 0: self.overlap += 1
            if self.ecc == True:
                fhead += file.read(4)
                self.ecc_dsize = struct.unpack('>B', fhead[0xc:0xd])[0]
                self.ecc_codesize = struct.unpack('>B', fhead[0xd:0xe])[0]
                self.crc = fhead[0xe:0x10]                             # 0x0e-2B: ANSI CRC16 of Audio Data

        if self.frmbytes == variables.FRM_MAXSZ:
            fhead += file.read(8)
            self.frmbytes = struct.unpack('>Q', fhead[-8:])[0]

        self.headlen = len(fhead)
        return False

filelist = []

@atexit.register
def cleanup():
    for file, _, _ in filelist:
        try:
            if os.path.exists(file): os.remove(file)
        except: pass

class decode:
    @staticmethod
    def overlap(frame: np.ndarray, overlap_fragment: np.ndarray, asfh: ASFH) -> tuple[np.ndarray, np.ndarray]:
        if overlap_fragment.shape != np.array([]).shape:
            fade_in = np.linspace(0, 1, len(overlap_fragment))
            fade_out = np.linspace(1, 0, len(overlap_fragment))
            for c in range(asfh.chnl):
                frame[:len(overlap_fragment), c] = \
                (frame[:len(overlap_fragment), c]  * fade_in) +\
                (overlap_fragment[:, c] * fade_out)
        next_overlap = np.array([])
        if asfh.profile in profiles.COMPACT and asfh.overlap != 0:
            olap = min(max(asfh.overlap, 2), 256)
            next_overlap = frame[(len(frame) * (olap - 1)) // olap:]
            frame = frame[:-len(next_overlap)]
        return frame, next_overlap

    @staticmethod
    def write(frame: np.ndarray, playstream: sd.OutputStream, filestream: io.BufferedWriter, dtype: str, play: bool, ispipe: bool) -> None:
        if frame.shape != np.array([]).shape:
            if play: playstream.write(frame.astype(np.float32))
            else:
                dt, dp = methods.get_dtype(dtype)
                if not dtype.startswith('f'):
                    if dtype.startswith('u'): frame+=1
                    frame *= 2**(dp*8-1)
                if ispipe: sys.stdout.buffer.write(frame.astype(dt).tobytes())
                else: filestream.write(frame.astype(dt).tobytes())
        return None

    @staticmethod
    def internal(file_path: str, **kwargs) -> None:
        speed: float = kwargs.get('speed', 1)
        play: bool = kwargs.get('play', False)
        ispipe: bool = kwargs.get('pipe', False)
        fix_error: bool = kwargs.get('ecc', False)
        gain: float = kwargs.get('gain', 1)
        verbose: bool = kwargs.get('verbose', False)
        dtype: str = kwargs.get('dtype', 'f64be')
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
            srate = int()

            # Container starts with 'fRad', and Stream starts with 0xffd0d297
            if ftype == 'container':
                head_len = struct.unpack('>Q', head[0x8:0x10])[0] # 0x08-8B: Total header size
            elif ftype == 'stream': head_len = 0
            else: raise ValueError(f'Invalid file signature: {ftype}')
            f.seek(head_len)
            t_accr, ddict = dict(), dict()
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
                if fhead != variables.FRM_SIGN:
                    hq = f.read(1)
                    if not hq:
                        if asfh.profile in profiles.COMPACT and asfh.overlap != 0: ddict[asfh.srate] += asfh.fsize//asfh.overlap
                        break
                    fhead = fhead[1:]+hq
                    continue
                fhead = None
                force_flush = asfh.update(f)
                if force_flush: continue
                data = f.read(asfh.frmbytes)
                if fix_error:
                    if ((asfh.profile in profiles.LOSSLESS and zlib.crc32(data) != struct.unpack('>I', asfh.crc)[0])
                    or  (asfh.profile in profiles.COMPACT and asfh.ecc and methods.crc16_ansi(data) != struct.unpack('>H', asfh.crc)[0])
                    ):
                        error_dir.append(str(framescount))
                        if not warned: warned = True; terminal("This file may had been corrupted. Please repack your file via 'ecc' option for the best music experience.")

                try: ddict[asfh.srate] += asfh.fsize
                except: ddict[asfh.srate] = asfh.fsize
                if asfh.profile in profiles.COMPACT and asfh.overlap != 0:
                    ddict[asfh.srate] -= asfh.fsize - (asfh.fsize * (asfh.overlap - 1)) // asfh.overlap

                dlen += asfh.frmbytes
                framescount += 1

            # show error frames
            if error_dir != []: terminal(f'Corrupt frames: {", ".join(error_dir)}')
            duration = sum([ddict[k] / k for k in ddict]) / speed
            f.seek(head_len)

# ----------------------------------- Metadata ----------------------------------- #
# This block parses and shows the metadata and image data from the header.
# OPTIONAL: i don't even activate this block cuz it makes cli becomes messy

            # if verbose:
            #     meta, img = header.parse(file_path)
            #     if meta:
            #         meta_tlen = max([len(m[0]) for m in meta])
            #         terminal('Metadata')
            #         for m in meta:
            #             if '\n' in m[1]:
            #                 m[1] = m[1].replace('\n', f'\n{" "*max(meta_tlen+2, 19)}: ')
            #             terminal(f'  {m[0].ljust(17, ' ')}: {m[1]}')

# ----------------------------------- Decoding ----------------------------------- #
# This block decodes FrAD stream to PCM stream and writes it on stdout or a file.
# ESSENTIAL

            stdoutstrm = sd.OutputStream(channels=1)
            tempfstrm = open(os.devnull, 'wb')
            try:
                # Starting stream
                printed = False
                bps = bpstot = 0
                dlen = os.path.getsize(file_path) - head_len
                start_time = time.time()
                fhead, overlap_fragment, frame = None, np.array([]), np.array([])

    # ----------------------------- Main decode loop ----------------------------- #
                while True:
                    # Finding Audio Stream Frame Header(AFSH)
                    if fhead is None: fhead = f.read(4)
                    if fhead != variables.FRM_SIGN:
                        hq = f.read(1)
                        if not hq: decode.write(overlap_fragment, stdoutstrm, tempfstrm, dtype, play, ispipe); break
                        fhead = fhead[1:]+hq
                        continue

                    # Parsing ASFH & Reading Audio Stream Frame
                    fhead = None
                    force_flush = asfh.update(f)
                    if force_flush:
                        t_accr[srate*speed] += len(overlap_fragment)
                        bytes_accr += asfh.headlen
                        decode.write(overlap_fragment, stdoutstrm, tempfstrm, dtype, play, ispipe)
                        continue
                    data: bytes = f.read(asfh.frmbytes)

                    # Decoding ECC
                    if asfh.ecc:
                        if fix_error and ((asfh.profile in profiles.LOSSLESS and zlib.crc32(data)         != struct.unpack('>I', asfh.crc)[0])
                            or            (asfh.profile in profiles.COMPACT  and methods.crc16_ansi(data) != struct.unpack('>H', asfh.crc)[0])
                            ): data = ecc.decode(data, asfh.ecc_dsize, asfh.ecc_codesize)
                        else:  data = ecc.unecc( data, asfh.ecc_dsize, asfh.ecc_codesize)

                    # Decoding
                    frame: np.ndarray = fourier.digital(data, asfh.float_bits, asfh.chnl, asfh.endian, profile=asfh.profile, srate=asfh.srate, fsize=asfh.fsize) * gain

                    # if channels and sample rate changed
                    if channels != asfh.chnl or srate != asfh.srate:
                        channels, srate = asfh.chnl, asfh.srate
                        decode.write(overlap_fragment, stdoutstrm, tempfstrm, dtype, play, ispipe)
                        if play: stdoutstrm = sd.OutputStream(samplerate=int(asfh.srate*speed), channels=asfh.chnl); stdoutstrm.start()
                        else:
                            overlap_fragment = np.array([])
                            tempfstrm.close()
                            tempfstrm = open(tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.pcm').name, 'wb')
                            filelist.append([tempfstrm.name, channels, srate])

                    frame, overlap_fragment = decode.overlap(frame, overlap_fragment, asfh)

                    # Write PCM Stream
                    decode.write(frame, stdoutstrm, tempfstrm, dtype, play, ispipe)

# --------------------------- Verbose block, Optional ---------------------------- #
#
                    frameNo += 1
                    try: t_accr[srate*speed] += len(frame)
                    except: t_accr[srate*speed] = len(frame)
                    t_sec = sum([t_accr[k] / k for k in t_accr])
                    bytes_accr += asfh.frmbytes + asfh.headlen
                    if play:
                        bps = (((asfh.frmbytes+asfh.headlen) * 8) * asfh.srate / len(frame))
                        bpstot += bps
                        depth = variables.bit_depths[asfh.profile][asfh.float_bits]
                        lgs = int(math.log(asfh.srate, 1000))
                        lgv = int(math.log(bpstot/frameNo, 1000))
                        if verbose:
                            if printed: terminal(RM_CLI*5, end='')
                            terminal(f'{methods.tformat(t_sec)} / {methods.tformat(duration)} (Frame #{frameNo} / {framescount} Frame{(framescount!=1)*"s"})')
                            terminal(f'{depth}b@{asfh.srate/10**(lgs*3)} {['','k','M','G','T'][lgs]}Hz {not asfh.endian and"B"or"L"}E {asfh.chnl} channel{(asfh.chnl!=1)*"s"}')
                            lgf = int(math.log(bps, 1000))
                            terminal(f'Profile {asfh.profile}, ECC{asfh.ecc and f": {asfh.ecc_dsize}/{asfh.ecc_codesize}" or " disabled"}{asfh.profile in profiles.COMPACT and asfh.overlap != 0 and f", Overlap: 1/{asfh.overlap}" or ", Overlap: disabled" or ""}')
                            terminal(f'{len(frame)} sample{len(frame)!=1 and"s"or""}, {asfh.frmbytes} Byte{(asfh.frmbytes!=1)*"s"}({bps/10**(lgf*3):.3f} {['','k','M','G','T'][lgf]}bps) per frame')
                            terminal(f'{bpstot/frameNo/10**(lgv*3):.3f} {['','k','M','G','T'][lgv]}bps average')
                        else:
                            if printed: terminal(RM_CLI, end='')
                            cq = {1:'Mono',2:'Stereo',4:'Quad',6:'5.1 Surround',8:'7.1 Surround'}.get(asfh.chnl, f'{asfh.chnl} ch')
                            terminal(f'{methods.tformat(t_sec)} / {methods.tformat(duration)}, {asfh.profile in profiles.LOSSLESS and f"{depth}b@"or f"{bpstot/frameNo/10**(lgv*3):.3f} {['','k','M','G','T'][lgv]}bps "}{asfh.srate/10**(lgs*3)} {['','k','M','G','T'][lgs]}Hz {cq}')
                        printed = True
                    else:
                        if verbose:
                            elapsed_time = time.time() - start_time
                            bps = bytes_accr / elapsed_time
                            mult = t_sec / elapsed_time
                            printed = methods.logging(3, 'Decode', printed, percent=(bytes_accr*100/dlen), tbytes=bytes_accr, bps=bps, mult=mult, time=elapsed_time)
#
# ------------------------------- End verbose block ------------------------------ #

                stdoutstrm.stop()
                stdoutstrm.close()
                tempfstrm.close()
                if printed and play:
                    terminal(RM_CLI, end='')
                    if verbose: terminal(RM_CLI*4, end='')
            except KeyboardInterrupt:
                stdoutstrm.abort()
                stdoutstrm.close()
                tempfstrm.close()
                if not play:
                    terminal('Aborting...')
                sys.exit(0)

    @staticmethod
    def split_q(s) -> tuple[int|None, str]:
        if s == None: return None, 'c'
        if not s[0].isdigit(): terminal('Quality format should be [{Positive integer}{c/v/a}]'); return None, 'c'
        number = int(''.join(filter(str.isdigit, s)))
        strategy = ''.join(filter(str.isalpha, s))
        return number, strategy

    @staticmethod
    def setaacq(quality: int | None, channels: int):
        if quality == None:
            if channels == 1: return 256000
            elif channels == 2: return 320000
            else: return 160000 * channels
        return quality

    ffmpeg_lossless = ['wav', 'flac', 'wavpack', 'tta', 'truehd', 'alac', 'dts', 'mlp']

    @staticmethod
    def directcmd(temp_pcm, dtype, srate, channels, ffmpeg_cmd):
        command = [
            variables.ffmpeg, '-y',
            '-loglevel', 'error',
            '-f', dtype,
            '-ar', str(srate),
            '-ac', str(channels),
            '-i', temp_pcm,
            '-i', variables.meta,
        ]
        command.extend(['-map_metadata', '1', '-map', '0:a'])
        command.extend(ffmpeg_cmd)
        subprocess.run(command)

    @staticmethod
    def ffmpeg(temp_pcm, dtype, srate, channels, codec, f, s, out, ext, quality, strategy, new_srate):
        command = [
            variables.ffmpeg, '-y',
            '-loglevel', 'error',
            '-f', dtype,
            '-ar', str(srate),
            '-ac', str(channels),
            '-i', temp_pcm,
            '-i', variables.meta,
        ]
        if os.path.exists(f'{variables.meta}.image'):
            command.extend(['-i', f'{variables.meta}.image', '-c:v', 'copy'])

        command.extend(['-map_metadata', '1', '-map', '0:a'])

        if os.path.exists(f'{variables.meta}.image'):
            command.extend(['-map', '2:v'])

        if new_srate is not None and new_srate != srate: command.extend(['-ar', str(new_srate)])

        command.append('-c:a')
        if codec in ['wav', 'riff']:
            command.append(f'pcm_{f}')
        else: command.append(codec)

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
    def AppleAAC_macOS(temp_pcm, dtype, srate, channels, out, quality, strategy):
        try:
            quality = str(quality)
            command = [
                variables.ffmpeg, '-y',
                '-loglevel', 'error',
                '-f', dtype,
                '-ar', str(srate),
                '-ac', str(channels),
                '-i', temp_pcm,
                '-sample_fmt', 's32',
                '-f', 'flac', variables.temp_flac
            ]
            subprocess.run(command)
        except KeyboardInterrupt:
            terminal('Aborting...')
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
            terminal('Aborting...')
            sys.exit(0)

    @staticmethod
    def AppleAAC_Windows(temp_pcm, dtype, srate, channels, out, quality, new_srate):
        try:
            command = [
                variables.aac,
                '--raw', temp_pcm,
                '--raw-channels', str(channels),
                '--raw-rate', str(srate),
                '--raw-format', dtype[:-1],
                '--adts',
                '-c', str(quality),
            ]
            if new_srate is not None and new_srate != srate: command.extend(['--rate', str(new_srate)])
            command.extend([
                '-o', f'{out}.aac',
                '-s'
            ])
            subprocess.run(command)
        except KeyboardInterrupt:
            terminal('Aborting...')
            sys.exit(0)

    @staticmethod
    def dec(file_path, **kwargs):
        # FFmpeg Command
        ffmpeg_cmd = kwargs.get('directcmd', None)

        # Output file
        out: str = kwargs.get('out', None)

        # Output file specifications
        dtype: int = kwargs.get('dtype', 'f64be')
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
        decode.internal(file_path, ecc=ecc, gain=gain, pipe=(out=='pipe'and True or False), dtype=dtype, verbose=verbose)
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
                temp_pcm, channels, srate = filelist[z]

                if ffmpeg_cmd is not None:
                    decode.directcmd(temp_pcm, dtype, srate, channels, ffmpeg_cmd)
                else:
                    if (codec == 'aac' and srate <= 48000 and channels <= 2) or codec in ['appleaac', 'apple_aac']:
                        if strategy in ['c', 'a']: q = decode.setaacq(q, channels)
                        if platform.system() == 'Darwin': decode.AppleAAC_macOS(temp_pcm, dtype, srate, channels, (z==0 and out or f'{out}.{z}'), q, strategy)
                        elif platform.system() == 'Windows': decode.AppleAAC_Windows(temp_pcm, dtype, srate, channels, (z==0 and out or f'{out}.{z}'), q, new_srate)
                    elif codec not in ['pcm', 'raw']:
                        decode.ffmpeg(temp_pcm, dtype, srate, channels, codec, f, s, (z==0 and out or f'{out}.{z}'), ext, q, strategy, new_srate)
                    else:
                        shutil.move(temp_pcm, (z==0 and f'{out}.{ext}' or f'{out}.{z}.{ext}'))
                os.remove(temp_pcm)

        except KeyboardInterrupt: terminal('Aborting...')
        except Exception as exc: sys.exit(traceback.format_exc())
        finally: sys.exit(0)
