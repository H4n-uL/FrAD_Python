import os
import platform

class variables:
    nperseg = 2048

    dir = os.path.dirname(os.path.realpath(__file__))
    oper = platform.uname()
    arch = platform.machine().lower()

    if oper.system == 'Windows' and arch in ['amd64', 'x86_64']:
        ffmpeg = os.path.join(dir, 'res', 'codec', 'ffmpeg.Windows')
        ffprobe = os.path.join(dir, 'res', 'parser', 'ffprobe.Windows')
    elif oper.system == 'Darwin':
        ffmpeg = os.path.join(dir, 'res', 'codec', 'ffmpeg.macOS')
        ffprobe = os.path.join(dir, 'res', 'parser', 'ffprobe.macOS')
    else:
        if arch in ['amd64', 'x86_64']:
            ffmpeg = os.path.join(dir, 'res', 'codec', 'ffmpeg.AMD64')
            ffprobe = os.path.join(dir, 'res', 'parser', 'ffprobe.AMD64')
        if arch == 'arm64':
            ffmpeg = os.path.join(dir, 'res', 'codec', 'ffmpeg.AArch64')
            ffprobe = os.path.join(dir, 'res', 'parser', 'ffprobe.AArch64')

class methods:
    def signature(sign):
        if sign != b'\x16\xb0\x03':
            raise Exception('This is not Fourier Analogue file.')
