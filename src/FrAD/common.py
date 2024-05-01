import atexit, os, platform, subprocess, sys, tempfile

yd = 365.25
ys = yd * 86400

directory = os.path.dirname(os.path.realpath(__file__))
res = os.path.join(directory, 'res')

class variables:
    hash_block_size = 2**20

    temp =      tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.swv').name
    temp2 =     tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.swv').name
    temp_pcm =  tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.pcm').name
    temp_dsd =  tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.bstr').name
    temp_flac = tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.flac').name
    meta =      tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.meta').name

    resfiles = lambda x: [f for f in os.listdir(res) if x in f]
    if resfiles('ffmpeg'):  ffmpeg  = os.path.join(res, resfiles('ffmpeg')[0])
    else:                   ffmpeg  = 'ffmpeg'
    if resfiles('ffprobe'): ffprobe = os.path.join(res, resfiles('ffprobe')[0])
    else:                   ffprobe = 'ffprobe'

    oper = platform.uname()
    arch = platform.machine().lower()
    if   oper.system == 'Windows': aac = os.path.join(res, 'AppleAAC.Windows')
    elif oper.system == 'Darwin':  aac = 'afconvert'

    try:
        subprocess.run([ffmpeg,  '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([ffprobe, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if oper.system in ['Windows', 'Darwin']:
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
    def signature(sign):
        if sign != b'fRad':
            raise Exception('This is not Fourier Analogue file.')

    def cantreencode(sign):
        if sign == b'fRad':
            raise Exception('This is an already encoded Fourier Analogue file.')

    def tformat(n: float) -> str:
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

    @atexit.register
    def cleanup():
        temp_files = [
            variables.temp,
            variables.temp2,
            variables.temp_pcm,
            variables.temp_dsd,
            variables.temp_flac,
            variables.meta]

        for file in temp_files:
            if os.path.exists(file):
                try:os.remove(file)
                except:pass
