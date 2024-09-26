PIPEIN = ['pipe:', 'pipe:0', '-', '/dev/stdin', 'dev/fd/0']
PIPEOUT = ['pipe:', 'pipe:1', '-', '/dev/stdout', 'dev/fd/1']

import math, os, sys
from libfrad import StreamInfo

def format_time(n: float) -> str:
    if n < 0.0: return f'-{format_time(-n)}'
    julian = int(n / 31557600.0); n = n % 31557600.0
    days = int(n / 86400.0); n = n % 86400.0
    hours = int(n / 3600.0); n = n % 3600.0
    minutes = int(n / 60.0); n = n % 60.0

    if julian > 0: return f'J{julian}.{days:03d}:{hours:02d}:{minutes:02d}:{n:06.3f}'
    elif days > 0: return f'{days}:{hours:02d}:{minutes:02d}:{n:06.3f}'
    elif hours > 0: return f'{hours}:{minutes:02d}:{n:06.3f}'
    elif minutes > 0: return f'{minutes}:{n:06.3f}'
    elif n >= 1.0: return f'{n:.3f} s'
    elif n >= 0.001: return f'{n * 1000.0:.3f} ms'
    elif n >= 0.000001: return f'{n * 1000000.0:.3f} µs'
    elif n > 0.0: return f'{n * 1000000000.0:.3f} ns'
    else: return '0'

def format_bytes(n: float) -> str:
    if n < 1000.0: return f'{n}'
    exp = int(math.log10(n) // 3)
    unit = ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    return f'{n / 1000.0 ** exp:.3f} {unit[exp]}'

def format_speed(n: float) -> str:
    if n >= 100.0: return f'{n:.0f}'
    elif n >= 10.0: return f'{n:.1f}'
    elif n >= 1.0: return f'{n:.2f}'
    else: return f'{n:.3f}'

def logging(loglevel: int, log: StreamInfo, linefeed: bool):
    if loglevel == 0: return
    print(f'size={format_bytes(log.get_total_size())}B time={format_time(log.get_duration())} bitrate={format_bytes(log.get_bitrate())}bits/s speed={format_speed(log.get_speed())}x    ', end='\r', file=sys.stderr)
    if linefeed: print(file=sys.stderr)

def check_overwrite(writefile: str, overwrite: bool):
    if os.path.exists(writefile) and not overwrite:
        if sys.stdin.isatty():
            print('Output file already exists, overwrite? (Y/N)', file=sys.stderr)
            while True:
                print('> ', end='', file=sys.stderr)
                input_ = input().strip()
                if input_.lower() == 'y': break
                elif input_.lower() == 'n': print('Aborted.', file=sys.stderr); exit(0)
        else: print('Output file already exists, please provide --force(-y) flag to overwrite.', file=sys.stderr); exit(0)