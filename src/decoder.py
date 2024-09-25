from libfrad import Decoder
from libfrad.backend.pcmformat import ff_format_to_numpy_type
from common import PIPEIN, PIPEOUT, check_overwrite, logging
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

def decode(rfile: str, params: CliParams, play: bool):
    wfile_prim = params.output
    if rfile == '': print('Input file must be given'); exit(1)

    rpipe, wpipe = False, False

    if rfile in PIPEIN: rpipe = True
    elif not os.path.exists(rfile): print("Input file doesn't exist"); exit(1)
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

    sink = sd.OutputStream(dtype='float32')
    decoder = Decoder(params.enable_ecc)
    pcm_fmt = ff_format_to_numpy_type(params.pcm)

    no = 0
    while True:
        buf = readfile.read(32768)
        if not buf and decoder.is_empty(): break

        pcm, srate, critical_info_modified = decoder.process(buf)
        sink = write(play, writefile, sink, pcm, pcm_fmt, srate)
        logging(params.loglevel, decoder.streaminfo, False)

        if critical_info_modified and not wpipe:
            no += 1
            wfile = f'{wfile_prim}.{no}.pcm'; x = time.time()
            check_overwrite(wfile, params.overwrite)
            decoder.streaminfo.start_time += time.time() - x
            writefile = open(wfile, 'wb')

    pcm, srate, _ = decoder.flush()
    sink = write(play, writefile, sink, pcm, pcm_fmt, srate)
    logging(params.loglevel, decoder.streaminfo, True)
    if play: sink.close()
