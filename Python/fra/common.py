import base64, os, platform, secrets
import numpy as np
from scipy.interpolate import interp1d

class variables:
    hash_block_size = 2**20

    dir = os.path.dirname(os.path.realpath(__file__))
    temp = os.path.join(dir, f'temp.{base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.swv')
    temp2 = os.path.join(dir, f'temp.ecc.{base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.swv')
    temp_pcm = os.path.join(dir, f'temp.{base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.pcm')
    temp2_pcm = os.path.join(dir, f'temp.rate.{base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.pcm')
    temp_flac = os.path.join(dir, f'temp.{base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.flac')
    meta = os.path.join(dir, f'{base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.meta')

    oper = platform.uname()
    arch = platform.machine().lower()

    if oper.system == 'Windows' and arch in ['amd64', 'x86_64']:
        ffmpeg = os.path.join(dir, 'res', 'codec', 'ffmpeg.Windows')
        aac = os.path.join(dir, 'res', 'codec', 'AppleAAC.Windows')
        ffprobe = os.path.join(dir, 'res', 'parser', 'ffprobe.Windows')
    elif oper.system == 'Darwin':
        ffmpeg = os.path.join(dir, 'res', 'codec', 'ffmpeg.macOS')
        aac = 'afconvert'
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

    def resample_1sec(data, sr_origin, sr_new):
        len_orig = len(data)

        # Padding
        padding = sr_origin - len_orig
        if padding > 0:
            data = np.pad(data, (0, padding), 'constant')

        # Linear Prediction
        if len_orig >= 2:
            slope = (data[-1] - data[-2]) / (1 / sr_origin)
            predicted_value = data[-1] + slope / sr_origin
        else:
            predicted_value = data[-1]

        data = np.append(data, predicted_value)

        # Resampling
        new_length = (len_orig + padding) * (sr_new + 1) // sr_origin
        new_time_axis = np.linspace(0, (len_orig + padding) / sr_origin, new_length, endpoint=False)

        time_axis = np.linspace(0, (len_orig + padding) / sr_origin, len_orig + padding + 1)
        interpolator = interp1d(time_axis, data, kind='cubic')
        resampled_data = interpolator(new_time_axis)

        # Unpadding
        if padding > 0:
            padding_to_remove = int(padding / sr_origin * sr_new)
            return resampled_data[:-padding_to_remove]
        if padding == 0:
            resampled_data = resampled_data[:-1]

        return resampled_data
