import base64, os, platform, secrets, shutil, sys, traceback
import numpy as np
import scipy.signal as sps

class variables:
    hash_block_size = 2**20

    directory = os.path.dirname(os.path.realpath(__file__))
    tmpdir = os.path.join(directory, 'tempfiles')
    os.makedirs(tmpdir, exist_ok=True)
    temp =      os.path.join(tmpdir, f'temp.{  base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.swv')
    temp =      os.path.join(tmpdir, f'temp.{  base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.swv')
    temp2 =     os.path.join(tmpdir, f'temp.2.{base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.swv')
    temp_pcm =  os.path.join(tmpdir, f'temp.{  base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.pcm')
    temp2_pcm = os.path.join(tmpdir, f'temp.2.{base64.b64encode(secrets.token_bytes(64)).decode().replace("/", "_")}.pcm')
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
        if sign != b'\x16\xb0\x03':
            raise Exception('This is not Fourier Analogue file.')

    def resample(data, sr_origin, sr_new):
        return sps.resample(data, int(len(data) * sr_new / sr_origin))

    def resample_pcm(channels, sample_rate, new_sample_rate):
        if new_sample_rate is not None and int(new_sample_rate) != sample_rate:
            new_sample_rate = int(new_sample_rate)
            try:
                with open(variables.temp_pcm, 'rb') as samp_bef, open(variables.temp2_pcm, 'wb') as samp_aft:
                    while True:
                        block = samp_bef.read(4 * channels * sample_rate)
                        if not block: break
                        data_numpy = np.frombuffer(block, dtype=np.int32).astype(np.float64) / 2**31

                        freq = [data_numpy[i::channels] for i in range(channels)]
                        block = (np.column_stack([methods.resample(c, sample_rate, new_sample_rate) for c in freq])
                                 .ravel(order='C') * 2**31).astype(np.int32)
                        samp_aft.write(block.tobytes())
                shutil.move(variables.temp2_pcm, variables.temp_pcm)
            except Exception as e:
                os.remove(variables.temp_pcm)
                os.remove(variables.temp2_pcm)
                if type(e) == KeyboardInterrupt:
                    sys.exit(0)
                else:
                    print(traceback.format_exc())
                    sys.exit(1)
            return new_sample_rate
        return sample_rate
