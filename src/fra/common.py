import base64, os, platform, secrets, shutil, subprocess

class variables:
    hash_block_size = 2**20

    directory = os.path.dirname(os.path.realpath(__file__))
    tmpdir = os.path.join(directory, 'tempfiles')
    os.makedirs(tmpdir, exist_ok=True)
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
        if sign != b'fRad':
            raise Exception('This is not Fourier Analogue file.')

    def resample_pcm(channels, sample_rate, new_sample_rate):
        if new_sample_rate is not None and new_sample_rate != sample_rate:
            command = [
                variables.ffmpeg,
                '-v', 'quiet',
                '-f', 's32le',
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-i', variables.temp_pcm,
                '-ar', str(new_sample_rate),
                '-f', 's32le',
                variables.temp2_pcm]
            subprocess.run(command)
            shutil.move(variables.temp2_pcm, variables.temp_pcm)
            return new_sample_rate
        return sample_rate
