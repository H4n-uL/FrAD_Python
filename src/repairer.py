from libfrad import Repairer
try:
    from .common import PIPEIN, PIPEOUT, check_overwrite, format_si, format_speed, format_time
    from .tools.cli import CliParams
    from .tools.process import ProcessInfo
except ImportError:
    from common import PIPEIN, PIPEOUT, check_overwrite, format_si, format_speed, format_time
    from tools.cli import CliParams
    from tools.process import ProcessInfo
import os, sys, time

def logging_repair(loglevel: int, log: ProcessInfo, linefeed: bool):
    if loglevel == 0: return
    
    print(f'size={format_si(log.get_total_size())}B speed={format_si(log.get_total_size() / (time.time() - log.start_time))}B/s    ', end='\r', file=sys.stderr)
    if linefeed: print(file=sys.stderr)

def repair(rfile: str, params: CliParams):
    wfile = params.output
    if rfile == '': print('Input file must be given'); exit(1)

    rpipe, wpipe = False, False

    if rfile in PIPEIN: rpipe = True
    elif not os.path.exists(rfile): print("Input file doesn't exist"); exit(1)
    if wfile in PIPEOUT: wpipe = True
    elif not rpipe and os.path.exists(wfile) and os.path.samefile(rfile, wfile): print('Input and wfile files cannot be the same'); exit(1)

    if wfile == '':
        wfrf = os.path.basename(rfile).split('.')
        wfile = f'{".".join(wfrf[:-1])}.recov.{wfrf[-1]}'

    check_overwrite(wfile, params.overwrite)

    readfile = open(rfile, 'rb') if not rpipe else sys.stdin.buffer
    writefile = open(wfile, 'wb') if not wpipe else sys.stdout.buffer

    repairer = Repairer(params.ecc_ratio)
    procinfo = ProcessInfo()

    while True:
        buf = readfile.read(32768)
        if not buf and repairer.is_empty(): break

        repaired = repairer.process(buf)
        procinfo.update(len(repaired), 0, 0)
        writefile.write(repaired)
        logging_repair(params.loglevel, procinfo, False)

    repaired = repairer.flush()
    procinfo.update(len(repaired), 0, 0)
    writefile.write(repaired)
    logging_repair(params.loglevel, procinfo, True)
