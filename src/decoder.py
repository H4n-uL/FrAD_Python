from libfrad import Decoder, ASFH, ProcessInfo, BIT_DEPTHS, ff_format_to_numpy_type
try:
    from .common import PIPEIN, PIPEOUT, check_overwrite, format_bytes, format_time, format_speed
    from .tools.cli import CliParams
except ImportError:
    from common import PIPEIN, PIPEOUT, check_overwrite, format_bytes, format_time, format_speed
    from tools.cli import CliParams
import io, os, sys, time
import sounddevice as sd
from typing import BinaryIO
import numpy as np

EMPTY = np.array([]).shape

def write(play: bool, writefile: io.BufferedWriter | BinaryIO, sink: sd.OutputStream, pcm: np.ndarray, fmt: np.dtype, srate: int) -> sd.OutputStream:
    if pcm.shape == EMPTY: return sink
    if play:
        if sink.samplerate != srate or sink.channels != len(pcm[0]):
            sink.close(); sink = sd.OutputStream(samplerate=srate, channels=len(pcm[0]), dtype='float32'); sink.start()
        sink.write(pcm.astype('float32'))
    else: writefile.write(pcm.astype(fmt).tobytes())

    return sink

def logging_decode(loglevel: int, procinfo: ProcessInfo, linefeed: bool, asfh: ASFH):
    if loglevel == 0: return

    out = []

    out.append(f'size={format_bytes(procinfo.get_total_size())}B time={format_time(procinfo.get_duration())} bitrate={format_bytes(procinfo.get_bitrate())}bits/s speed={format_speed(procinfo.get_speed())}x    ')
    if loglevel > 1: out.append(f'Profile {asfh.profile}, {BIT_DEPTHS[asfh.profile][asfh.bit_depth_index]}bits {asfh.channels}ch@{asfh.srate}Hz, ECC={"disabled" if not asfh.ecc else f"{asfh.ecc_dsize}/{asfh.ecc_codesize}"}    ')

    line_count = len(out) - 1
    print('\n'.join(out), end='', file=sys.stderr)

    if linefeed: print(file=sys.stderr)
    else:
        for _ in range(line_count): print('\x1b[1A', end='', file=sys.stderr)
        print('\r', end='', file=sys.stderr)

def decode(rfile: str, params: CliParams, play: bool):
    wfile_prim = params.output
    if rfile == '': print('Input file must be given', file=sys.stderr); exit(1)

    rpipe, wpipe = False, False

    if rfile in PIPEIN: rpipe = True
    elif not os.path.exists(rfile): print("Input file doesn't exist", file=sys.stderr); exit(1)
    if wfile_prim in PIPEOUT or play: wpipe = True
    elif not (rpipe or play) and os.path.exists(wfile_prim) and os.path.samefile(rfile, wfile_prim): print('Input and Output files cannot be the same'); exit(1)

    if wfile_prim == '':
        wfrf = os.path.basename(rfile)
        wfile_prim = '.'.join(wfrf.split('.')[:-1])
    elif wfile_prim.endswith('.pcm'): wfile_prim = wfile_prim[:-4]

    wfile = f'{wfile_prim}.pcm'
    if not wpipe: check_overwrite(wfile, params.overwrite)
    
    readfile = open(rfile, 'rb') if not rpipe else sys.stdin.buffer
    writefile = open(wfile, 'wb') if not wpipe else sys.stdout.buffer
    if play: params.loglevel = 0

    sink = sd.OutputStream(samplerate=48000, channels=1, dtype='float32')
    params.speed = params.speed if params.speed > 0 else 1.0
    decoder = Decoder(params.enable_ecc)
    pcm_fmt = ff_format_to_numpy_type(params.pcm)

    frames, no = 0, 0
    while True:
        bufsize = 32768
        if frames > 0 and play: bufsize = decoder.procinfo.get_total_size() // frames
        buf = readfile.read(bufsize)
        if not buf and decoder.is_empty(): break

        decoded = decoder.process(buf)
        sink = write(play, writefile, sink, decoded.pcm, pcm_fmt, int(decoded.srate * params.speed))
        logging_decode(params.loglevel, decoder.procinfo, False, decoder.get_asfh())

        if decoded.crit and not wpipe:
            no += 1; wfile = f'{wfile_prim}.{no}.pcm'
            decoder.procinfo.block()
            check_overwrite(wfile, params.overwrite)
            decoder.procinfo.unblock()
            writefile = open(wfile, 'wb')

        frames += decoded.frames

    decoded = decoder.flush()
    sink = write(play, writefile, sink, decoded.pcm, pcm_fmt, int(decoded.srate * params.speed))
    logging_decode(params.loglevel, decoder.procinfo, True, decoder.get_asfh())
    if play: sink.close()
