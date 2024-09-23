from libfrad import Decoder
from libfrad.backend.pcmformat import ff_format_to_numpy_type
from common import PIPEIN, PIPEOUT, check_overwrite, logging
from tools.cli import CliParams
import io, os, sys
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
    wfile = params.output
    if rfile == '': print("Input file must be given"); exit(1)

    rpipe, wpipe = False, False

    if rfile in PIPEIN: rpipe = True
    elif not os.path.exists(rfile): print("Input file doesn't exist"); exit(1)
    if wfile in PIPEOUT: wpipe = True
    elif not (rpipe or play) and os.path.exists(wfile) and os.path.samefile(rfile, wfile): print("Input and wfile files cannot be the same"); exit(1)

    if wfile == '':
        wfrf = os.path.basename(rfile)
        wfile = '.'.join(wfrf.split('.')[:-1])
    elif wfile.endswith('.pcm'): wfile = wfile[:-4]

    check_overwrite(wfile, params.overwrite)
    
    readfile = open(rfile, 'rb') if not rpipe else sys.stdin.buffer
    writefile = open(f"{wfile}.pcm", 'wb') if not wpipe and not play else sys.stdout.buffer
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

        if critical_info_modified and not (wpipe or play):
            no += 1; writefile = open(f"{wfile}.{no}.pcm", 'wb')

    pcm, srate, _ = decoder.flush()
    sink = write(play, writefile, sink, pcm, pcm_fmt, srate)
    logging(params.loglevel, decoder.streaminfo, True)
    if play: sink.close()
