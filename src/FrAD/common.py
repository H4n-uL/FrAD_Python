import atexit, os, platform, subprocess, sys, tempfile

yd = 365.25
ys = yd * 86400

class variables:
    hash_block_size = 2**20

    directory = os.path.dirname(os.path.realpath(__file__))

    temp =      tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.swv').name
    temp2 =     tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.swv').name
    temp_pcm =  tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.pcm').name
    temp_dsd =  tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.bstr').name
    temp_flac = tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.flac').name
    meta =      tempfile.NamedTemporaryFile(prefix='frad_', delete=True, suffix='.meta').name

    @atexit.register
    def cleanup():
        try:
            if os.path.exists(variables.temp):      os.remove(variables.temp)
            if os.path.exists(variables.temp2):     os.remove(variables.temp2)
            if os.path.exists(variables.temp_pcm):  os.remove(variables.temp_pcm)
            if os.path.exists(variables.temp_dsd):  os.remove(variables.temp_dsd)
            if os.path.exists(variables.temp_flac): os.remove(variables.temp_flac)
            if os.path.exists(variables.meta):      os.remove(variables.meta)
        except:
            if os.path.exists(variables.temp):      os.remove(variables.temp)
            if os.path.exists(variables.temp2):     os.remove(variables.temp2)
            if os.path.exists(variables.temp_pcm):  os.remove(variables.temp_pcm)
            if os.path.exists(variables.temp_dsd):  os.remove(variables.temp_dsd)
            if os.path.exists(variables.temp_flac): os.remove(variables.temp_flac)
            if os.path.exists(variables.meta):      os.remove(variables.meta)
            sys.exit(0)

    oper = platform.uname()
    arch = platform.machine().lower()

    if getattr(sys, 'frozen', False):
        ffmpeg = 'ffmpeg'
        ffprobe = 'ffprobe'
        if oper.system == 'Windows':
            aac = os.path.join(directory, 'qaac.exe')
        elif oper.system == 'Darwin':
            aac = 'afconvert'
        try:
            subprocess.run([ffmpeg, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run([ffprobe, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("Error: ffmpeg or ffprobe not found. Please install them and try again.")
            sys.exit(1)

    else:
        if oper.system == 'Windows' and arch in ['amd64', 'x86_64']:
            ffmpeg =      os.path.join(directory, 'res', 'codec',  'ffmpeg.Windows')
            aac =         os.path.join(directory, 'res', 'codec',  'AppleAAC.Windows')
            ffprobe =     os.path.join(directory, 'res', 'parser', 'ffprobe.Windows')
        elif oper.system == 'Darwin':
            ffmpeg =      os.path.join(directory, 'res', 'codec',  'ffmpeg.macOS')
            aac =         'afconvert'
            ffprobe =     os.path.join(directory, 'res', 'parser', 'ffprobe.macOS')
        else:
            if arch in ['amd64', 'x86_64']:
                ffmpeg =  os.path.join(directory, 'res', 'codec',  'ffmpeg.AMD64')
                ffprobe = os.path.join(directory, 'res', 'parser', 'ffprobe.AMD64')
            if arch == 'arm64':
                ffmpeg =  os.path.join(directory, 'res', 'codec',  'ffmpeg.AArch64')
                ffprobe = os.path.join(directory, 'res', 'parser', 'ffprobe.AArch64')

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
