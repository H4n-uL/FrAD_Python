import atexit, math, os, platform, subprocess, sys, tempfile, tarfile

yd = 365.25
ys = yd * 86400

directory = os.path.dirname(os.path.realpath(__file__))
res = os.path.join(directory, 'res')

class variables:
    cli_width = 40
    overlap_rate = 16

    # Temporary files for metadata processing / stream repairing
    temp =      tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.frad').name
    temp2 =     tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.frad').name
    # PCM to DSD conversion
    temp_dsd =  tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.bstr').name
    # PCM -> FLAC -> AAC conversion for afconvert AppleAAC
    temp_flac = tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.flac').name
    # ffmeta
    meta =      tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.meta').name

    # Finding ffmpeg and ffprobe from src/FrAD/res
    resfiles = lambda x: [f for f in os.listdir(res) if x in f]
    if resfiles('ffmpeg'):  ffmpeg  = os.path.join(res, resfiles('ffmpeg')[0])
    else:                   ffmpeg  = 'ffmpeg'
    if resfiles('ffprobe'): ffprobe = os.path.join(res, resfiles('ffprobe')[0])
    else:                   ffprobe = 'ffprobe'

    # Setting up AppleAAC encoder for each platforms
    oper = platform.uname()
    arch = platform.machine().lower()
    if oper.system == 'Windows':
        AppleAAC_win = os.path.join(res, 'AppleAAC.Win.tar.gz')
        aac = os.path.join(res, 'AppleAAC.Windows')
        if os.path.isfile(AppleAAC_win) and not os.path.isfile(aac): tarfile.open(AppleAAC_win, 'r:gz').extractall(path=res)
    elif oper.system == 'Darwin':  aac = 'afconvert'
    else:                          aac = None

    # ffmpeg and ffprobe installation verification
    try:
        subprocess.run([ffmpeg,  '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([ffprobe, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if aac is not None:
            subprocess.run([aac,       '-h'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print('Error: ffmpeg or ffprobe not found. Please install and try again,')
        print(f'or download and put them in {res}')
        if oper.system == 'Windows':  print('QAAC is built-in on this repository.')
        elif oper.system == 'Darwin': print('afconvert is built-in on macOS')
        else:
            print('On Linux, you have no way to use Apple AAC encoder.')
            print('can anyone please reverse-engineer it and open its source')
        sys.exit(1)

class methods:
    @staticmethod
    def signature(sign: bytes): # Verifying FrAD signature
        if sign == b'fRad': return 'container'
        elif sign == b'\xff\xd0\xd2\x97': return 'stream'
        else: raise Exception('This is not Fourier Analogue-in-Digital.')

    @staticmethod
    def cantreencode(sign: bytes): # Anti-re-encode
        if sign in [b'fRad', b'\xff\xd0\xd2\x97']:
            raise Exception('This is an already encoded Fourier Analogue file.')

    @staticmethod
    def tformat(n: float | str) -> str: # i'm dying help
        if type(n) != float: return str(n)
        if n < 0: return f'-{methods.tformat(-n)}'
        if n == 0: return '0'
        if n < 0.000001: return f'{n*10**9:.3f} ns'
        if n < 0.001: return f'{n*10**6:.3f} µs'
        if n < 1: return f'{n*1000:.3f} ms'
        if n < 60: return f'{n:.3f} s'
        if n < 3600: return f'{int(n//60)%60}:{n%60:06.3f}'
        if n < 86400: return f'{int(n//3600)%24}:{int(n//60)%60:02d}:{n%60:06.3f}'
        if n < ys: return f'{int(n//86400)%yd}:{int(n//3600)%24:02d}:{int(n//60)%60:02d}:{n%60:06.3f}'
        return f'J{int(n//ys)}.{int((n%ys//86400)%yd):03d}:{int(n%ys//3600)%24:02d}:{int(n%ys//60)%60:02d}:{n%ys%60:06.3f}'

    @staticmethod
    def get_dtype(raw: str | None) -> tuple[str, int]:
        # in [s,u,f]{bit depth}[be,le], e.g. s16be, u32le, f64be
        # This is only for Numpy dtype, and should be implemented differently for each implementations
        if not raw: return '>f8', 8
        if raw[-2:] in ['be', 'le']:
            raw, endian = raw[:-2], raw[-2:]
            endian = endian=='be' and '>' or endian=='le' and '<' or ''
        else: endian = ''
        raw, ty = raw[1:], raw[0]
        if ty=='s': ty = 'i'
        elif ty=='u': ty = 'u'
        elif ty=='f': ty = 'f'
        else: print(f'Invalid raw PCM type: {ty}'); sys.exit(1)
        depth = int(raw)//8
        return f'{endian}{ty}{depth}', depth

    @atexit.register
    def cleanup():
        temp_files = [
            variables.temp,
            variables.temp2,
            variables.temp_dsd,
            variables.temp_flac,
            variables.meta]

        for file in temp_files:
            if os.path.exists(file):
                try:os.remove(file)
                except:pass
    
    @staticmethod
    def logging(loglevel: int, method: str, printed: bool, **kwargs) -> bool:
        if loglevel == 1:
            percent = kwargs.get('percent', 0)
            prgbar = int(percent / 100 * variables.cli_width)
            if printed: print('\x1b[1A\x1b[2K', end='')
            print(f"[{'█'*prgbar}{' '*(variables.cli_width-prgbar)}] {percent:.3f}% completed")
        elif loglevel == 2:
            percent = kwargs.get('percent', 0)
            bps = kwargs.get('bps', 0)
            elapsed_time = kwargs.get('time', 0)
            mult = kwargs.get('mult', None)

            lgb = int(math.log(bps, 1000))
            prgbar = int(percent / 100 * variables.cli_width)
            eta = (elapsed_time / (percent / 100)) - elapsed_time if percent != 0 else 'infinity'
            if printed: print('\x1b[1A\x1b[2K'*3, end='')
            print(f'{method} speed: {(bps/10**(lgb*3)):.3f} {['','k','M','G','T'][lgb]}B/s{mult and f", X{mult:.3f}" or ""}')
            print(f'elapsed: {methods.tformat(elapsed_time)}, ETA {methods.tformat(eta)}')
            print(f"[{'█'*prgbar}{' '*(variables.cli_width-prgbar)}] {percent:.3f}% completed")
        return True