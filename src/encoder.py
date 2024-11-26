from libfrad import Encoder, profiles, head
try:
    from .common import PIPEIN, PIPEOUT, check_overwrite, format_si, format_speed, format_time
    from .tools.cli import CliParams
    from .tools.process import ProcessInfo
except ImportError:
    from common import PIPEIN, PIPEOUT, check_overwrite, format_si, format_speed, format_time
    from tools.cli import CliParams
    from tools.process import ProcessInfo
from typing import BinaryIO
import io, os, sys

def set_files(rfile: str, wfile: str, profile: int, overwrite: bool) -> tuple[io.BufferedReader | BinaryIO, io.BufferedWriter | BinaryIO]:
    rpipe, wpipe = False, False

    if rfile in PIPEIN: rpipe = True
    elif not os.path.exists(rfile): print("Input file doesn't exist"); exit(1)
    if wfile in PIPEOUT: wpipe = True
    elif not rpipe and os.path.exists(wfile) and os.path.samefile(rfile, wfile): print('Input and wfile files cannot be the same'); exit(1)

    if wfile == '':
        wfrf = os.path.basename(rfile)
        wfile = '.'.join(wfrf.split('.')[:-1])

    if not (wfile.endswith('.frad') or wfile.endswith('.dsin') or wfile.endswith('.fra') or wfile.endswith('.dsn')):
        if profile in profiles.LOSSLESS:
            if len(wfile) <= 8: wfile = f'{wfile}.fra'
            else: wfile = f'{wfile}.frad'
        elif len(wfile) <= 8: wfile = f'{wfile}.dsn'
        else: wfile = f'{wfile}.dsin'

    check_overwrite(wfile, overwrite)

    readfile = open(rfile, 'rb') if not rpipe else sys.stdin.buffer
    writefile = open(wfile, 'wb') if not wpipe else sys.stdout.buffer

    return readfile, writefile

def logging_encode(loglevel: int, log: ProcessInfo, linefeed: bool):
    if loglevel == 0: return
    print(f'size={format_si(log.get_total_size())}B time={format_time(log.get_duration())} bitrate={format_si(log.get_bitrate())}bits/s speed={format_speed(log.get_speed())}x    ', end='\r', file=sys.stderr)
    if linefeed: print(file=sys.stderr)

def encode(input: str, params: CliParams):
    if input == '': print('Input file must be given', file=sys.stderr); exit(1)

    encoder = Encoder(params.profile, params.pcm)
    if params.srate == 0: print('Sample rate should be set except zero', file=sys.stderr); exit(1)
    if params.channels == 0: print('Channel count should be set except zero', file=sys.stderr); exit(1)

    encoder.set_srate(params.srate)
    encoder.set_channels(params.channels)
    encoder.set_frame_size(params.frame_size)
    encoder.set_ecc(params.enable_ecc, params.ecc_ratio)
    encoder.set_little_endian(params.little_endian)
    encoder.set_bit_depth(params.bits)
    encoder.set_overlap_ratio(params.overlap_ratio)
    loss_level = 1.25 ** params.losslevel / 19.0 + 0.5
    encoder.set_loss_level(loss_level)

    rfile, wfile = set_files(input, params.output, params.profile, params.overwrite)

    image = open(params.image_path, 'rb').read() if params.image_path != '' and os.path.exists(params.image_path) else b''
    wfile.write(head.builder(params.meta, image))

    procinfo = ProcessInfo()
    while True:
        pcm_buf = rfile.read(32768)
        if not pcm_buf: break

        encoded = encoder.process(pcm_buf)
        procinfo.update(len(encoded.buf), encoded.samples, encoder.get_srate())
        wfile.write(encoded.buf)
        logging_encode(params.loglevel, procinfo, False)

    encoded = encoder.flush()
    procinfo.update(len(encoded.buf), encoded.samples, encoder.get_srate())
    wfile.write(encoded.buf)
    logging_encode(params.loglevel, procinfo, True)
