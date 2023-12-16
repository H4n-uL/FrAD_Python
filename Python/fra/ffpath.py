import os
import platform

class ff:
    dir = os.path.dirname(os.path.realpath(__file__))
    oper = platform.uname()
    arch = platform.machine().lower()

    if oper.system == 'Windows' and arch in ['amd64', 'x86_64']:
        mpeg = os.path.join(dir, 'tools', 'ffmpeg.Windows')
        probe = os.path.join(dir, 'tools', 'ffprobe.Windows')
    elif oper.system == 'Darwin':
        mpeg = os.path.join(dir, 'tools', 'ffmpeg.macOS')
        probe = os.path.join(dir, 'tools', 'ffprobe.macOS')
    else:
        if arch in ['amd64', 'x86_64']:
            mpeg = os.path.join(dir, 'tools', 'ffmpeg.AMD64')
            probe = os.path.join(dir, 'tools', 'ffprobe.AMD64')
        if arch == 'arm64':
            mpeg = os.path.join(dir, 'tools', 'ffmpeg.AArch64')
            probe = os.path.join(dir, 'tools', 'ffprobe.AArch64')
