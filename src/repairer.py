from libfrad import Repairer
from common import PIPEIN, PIPEOUT, check_overwrite, logging
from tools.cli import CliParams
import os, sys

def repair(rfile: str, params: CliParams):
    wfile = params.output
    if rfile == '': print("Input file must be given"); exit(1)

    rpipe, wpipe = False, False

    if rfile in PIPEIN: rpipe = True
    elif not os.path.exists(rfile): print("Input file doesn't exist"); exit(1)
    if wfile in PIPEOUT: wpipe = True
    elif not rpipe and os.path.exists(wfile) and os.path.samefile(rfile, wfile): print("Input and wfile files cannot be the same"); exit(1)

    if wfile == '':
        wfrf = os.path.basename(rfile).split('.')
        wfile = f'{".".join(wfrf[:-1])}.recov.{wfrf[-1]}'

    check_overwrite(wfile, params.overwrite)

    readfile = open(rfile, 'rb') if not rpipe else sys.stdin.buffer
    writefile = open(wfile, 'wb') if not wpipe else sys.stdout.buffer

    repairer = Repairer(params.ecc_ratio)
    while True:
        buf = readfile.read(32768)
        if not buf and repairer.is_empty(): break

        writefile.write(repairer.process(buf))
        logging(params.loglevel, repairer.procinfo, False)
    writefile.write(repairer.flush())
    logging(params.loglevel, repairer.procinfo, True)
