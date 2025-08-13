PIPEIN = ['pipe:', 'pipe:0', '-', '/dev/stdin', 'dev/fd/0']
PIPEOUT = ['pipe:', 'pipe:1', '-', '/dev/stdout', 'dev/fd/1']

import math, os, sys

def get_file_stem(file: str) -> str:
    if file in PIPEIN or file in PIPEOUT: return 'pipe'
    base = os.path.basename(file)
    if (base.startswith('.') and base.count('.') == 1) or base.count('.') == 0: return base
    return '.'.join(base.split('.')[:-1])

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

def format_si(n: float) -> str:
    if n == 0: return '0 '
    exp = int(math.log10(n) // 3)
    unit = ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    return f'{n / 1000.0 ** exp:.3f} {unit[exp]}'

def format_speed(n: float) -> str:
    if n >= 100.0: return f'{n:.0f}'
    elif n >= 10.0: return f'{n:.1f}'
    elif n >= 1.0: return f'{n:.2f}'
    else: return f'{n:.3f}'

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