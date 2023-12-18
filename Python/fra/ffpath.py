import os
import platform

class ff:
    dir = os.path.dirname(os.path.realpath(__file__))
    oper = platform.uname()
    arch = platform.machine().lower()

    if oper.system == 'Windows' and arch in ['amd64', 'x86_64']:
        mpeg = os.path.join(dir, 'res', 'codec', 'ffmpeg.Windows')
        probe = os.path.join(dir, 'res', 'parser', 'ffprobe.Windows')
    elif oper.system == 'Darwin':
        mpeg = os.path.join(dir, 'res', 'codec', 'ffmpeg.macOS')
        probe = os.path.join(dir, 'res', 'parser', 'ffprobe.macOS')
    else:
        if arch in ['amd64', 'x86_64']:
            mpeg = os.path.join(dir, 'res', 'codec', 'ffmpeg.AMD64')
            probe = os.path.join(dir, 'res', 'parser', 'ffprobe.AMD64')
        if arch == 'arm64':
            mpeg = os.path.join(dir, 'res', 'codec', 'ffmpeg.AArch64')
            probe = os.path.join(dir, 'res', 'parser', 'ffprobe.AArch64')
