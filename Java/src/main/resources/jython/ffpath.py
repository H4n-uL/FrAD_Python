import os
import platform

class ff:
    dir = os.path.dirname(os.path.realpath(__file__))
    arch = platform.machine()

    if os.name == 'nt':
        mpeg = os.path.join(dir, 'tools', 'ffmpeg.Windows')
        probe = os.path.join(dir, 'tools', 'ffprobe.Windows')
    elif os.name == 'posix':
        mpeg = os.path.join(dir, 'tools', 'ffmpeg.macOS')
        probe = os.path.join(dir, 'tools', 'ffprobe.macOS')
    else:
        if arch == 'AMD64':
            mpeg = os.path.join(dir, 'tools', 'ffmpeg.AMD64')
            probe = os.path.join(dir, 'tools', 'ffprobe.AMD64')
        if arch == 'arm64':
            mpeg = os.path.join(dir, 'tools', 'ffmpeg.AArch64')
            probe = os.path.join(dir, 'tools', 'ffprobe.AArch64')