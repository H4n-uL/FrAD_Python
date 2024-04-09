import base64, os, platform, secrets

yd = 365.25
ys = yd * 86400

class variables:
    hash_block_size = 2**20

    directory = os.path.dirname(os.path.realpath(__file__))
    tmpdir = os.path.join(directory, 'tempfiles')
    os.makedirs(tmpdir, exist_ok=True)
    temp =      os.path.join(tmpdir, f'temp.{  base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.swv')
    temp2 =     os.path.join(tmpdir, f'temp.2.{base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.swv')
    temp_pcm =  os.path.join(tmpdir, f'temp.{  base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.pcm')
    temp_dsd =  os.path.join(tmpdir, f'temp.{  base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.bitstream')
    temp_flac = os.path.join(tmpdir, f'temp.{  base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.flac')
    meta =      os.path.join(tmpdir, f'{       base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.meta')

    oper = platform.uname()
    arch = platform.machine().lower()

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

    def get_gain(glist):
        if glist[0] is None: return 1
        if glist[1] is True: return 10 ** (float(glist[0]) / 20)
        else: return float(glist[0])


    def tformat(n: float) -> str:
        if n < 0: return f'-{methods.tformat(-n)}'
        if n == 0: return '---'
        if n < 0.000001: return f'{n*10**9:.3f} ns'
        if n < 0.001: return f'{n*10**6:.3f} µs'
        if n < 1: return f'{n*1000:.3f} ms'
        if n < 60: return f'{n:.3f} s'
        if n < 3600: return f'{int(n//60)%60}:{n%60:06.3f}'
        if n < 86400: return f'{int(n//3600)%24}:{int(n//60)%60:02d}:{n%60:06.3f}'
        if n < ys: return f'{int(n//86400)%yd}:{int(n//3600)%24:02d}:{int(n//60)%60:02d}:{n%60:06.3f}'
        return f'J{int(n//ys)}.{int((n%ys//86400)%yd):03d}:{int(n%ys//3600)%24:02d}:{int(n%ys//60)%60:02d}:{n%ys%60:06.3f}'
