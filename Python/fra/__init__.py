from .encoder import encode
from .decoder import decode
from .header import header
from .player import player
from .repack import repack

import os
import platform

class fra:
    dir = os.path.dirname(os.path.realpath(__file__))
    arch = platform.machine()

    if os.name == 'nt':
        ffmpeg = os.path.join(dir, 'tools', 'ffmpeg.Windows')
        ffprobe = os.path.join(dir, 'tools', 'ffprobe.Windows')
    elif os.name == 'posix':
        ffmpeg = os.path.join(dir, 'tools', 'ffmpeg.macOS')
        ffprobe = os.path.join(dir, 'tools', 'ffprobe.macOS')
    else:
        if arch == 'AMD64':
            ffmpeg = os.path.join(dir, 'tools', 'ffmpeg.AMD64')
            ffprobe = os.path.join(dir, 'tools', 'ffprobe.AMD64')
        if arch == 'arm64':
            ffmpeg = os.path.join(dir, 'tools', 'ffmpeg.AArch64')
            ffprobe = os.path.join(dir, 'tools', 'ffprobe.AArch64')
