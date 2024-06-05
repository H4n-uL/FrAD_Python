import sys

encode_opt = ['encode', 'enc']
decode_opt = ['decode', 'dec']
meta_opt = ['meta', 'metadata']
repack_ecc_opt = ['ecc', 'repack']
play_opt = ['play']
record_opt = ['record', 'rec']
update_opt = ['update']

def terminal(*args: object, sep: str | None = ' ', end: str | None = '\n'):
    sys.stderr.buffer.write(f'{(sep or '').join(map(str,args))}{end}'.encode())
    sys.stderr.buffer.flush()

def parse_args(args: list[str]) -> tuple[str, str|None, str|None, dict]:
    try: action = args.pop(0)
    except: return '', '', '', dict()
    if action in meta_opt: metaoption = args.pop(0)
    else: metaoption = None
    file_path = None
    try: file_path = str(args.pop(0))
    except: pass
    options = {}

    while args:
        arg: str = args.pop(0)
        if arg.startswith('-'):
            key = arg.lstrip('-')

            # Output file path
            if key in ('o', 'output', 'out', 'output-file'):
                key, value = 'output', args.pop(0)

            # Bit depth
            elif key in ('b', 'bits', 'bit'):
                b = '<null>'
                try:
                    b = args.pop(0)
                    key, value = 'bits', int(b)
                except:
                    terminal(f'Value cannot be parsed as Integer: {arg} {b}')
                    sys.exit(1)

            # Image file path
            elif key in ('img', 'image'):
                key, value = 'image', args.pop(0)

            # New sample rate
            elif key in ('sr', 'srate', 'sample-rate', 'nsr', 'new-srate', 'new-sample-rate', 'resample'):
                nsr = '<null>'
                try:
                    nsr = args.pop(0)
                    key, value = 'srate', int(nsr)
                except:
                    terminal(f'Value cannot be parsed as Integer: {arg} {nsr}')
                    sys.exit(1)

            # Channels
            elif key in ('c', 'chnl', 'channel', 'channels'):
                chnl = '<null>'
                try:
                    chnl = args.pop(0)
                    key, value = 'chnl', int(chnl)
                except:
                    terminal(f'Value cannot be parsed as Integer: {arg} {chnl}')
                    sys.exit(1)

            # Samples per frame
            elif key in ('fr', 'fsize', 'frame-size', 'samples-per-frame'):
                fsz = '<null>'
                try:
                    fsz = args.pop(0)
                    key, value = 'fsize', int(fsz)
                except:
                    terminal(f'Value cannot be parsed as Integer: {arg} {fsz}')
                    sys.exit(1)

            # Overlap ratio
            elif key in ('olap', 'overlap'):
                olap = '<null>'
                try:
                    olap = args.pop(0)
                    key, value = 'overlap', int(olap)
                except:
                    terminal(f'Value cannot be parsed as Integer: {arg} {olap}')
                    sys.exit(1)

            # Codec type
            elif key in ('codec'):
                key, value = 'codec', args.pop(0)

            # Gain
            elif key in ('g', 'gain'):
                g_b = '<null>'
                try:
                    g = g_b = args.pop(0)
                    db = False
                    if g.lower().endswith('db'): g = g[:-2]; db = True
                    if g.lower().endswith('dbfs'): g = g[:-4]; db = True
                    if len(args) > 0 and args[0].lower() in ('db', 'dbfs'): db = True; args.pop(0)
                    key, value = 'gain', float(g)
                    if db: value = 10 ** (value / 20)
                except ValueError:
                    terminal(f'Value cannot be parsed as Float: {arg} {g_b}')
                    sys.exit(1)

            # Raw
            elif key in ('r', 'raw', 'pcm'):
                try: key, value = 'raw', args.pop(0)
                except ValueError:
                    terminal(f'Value cannot be parsed as String: {arg}')
                    sys.exit(1)

            # Enable ECC
            elif key in ('e', 'ecc', 'apply-ecc', 'enable-ecc'):
                key, value = 'ecc', True
                if len(args)!=0 and args[0].isdigit():
                    options[key] = value
                    d = e = '<null>'
                    try:
                        d = args.pop(0)
                        e = args.pop(0)
                        key, value = 'data-ecc', [int(d), int(e)]
                    except:
                        terminal(f'Value cannot be parsed as Integer: {arg} {d} {e}')
                        sys.exit(1)

            # Play speed
            elif key in ('spd', 'speed'):
                spd = '<null>'
                try:
                    spd = args.pop(0)
                    key, value = 'speed', float(spd)
                except:
                    terminal(f'Value cannot be parsed as Float: {arg} {spd}')
                    sys.exit(1)

            # Decode quality
            elif key in ('q', 'quality'):
                key, value = 'quality', args.pop(0)

            # Play keys
            elif key in ('k', 'keys', 'key'):
                k = '<null>'
                try:
                    k = args.pop(0)
                    key, value = 'keys', float(k)
                except:
                    terminal(f'Value cannot be parsed as Float: {arg} {k}')
                    sys.exit(1)

            # Metadata
            elif key in ('m', 'meta', 'metadata'):
                mk = mv = '<null>'
                try:
                    mk = args.pop(0)
                    mv = args.pop(0)
                    try: key, value = 'meta', options['meta']+[[mk, mv]]
                    except KeyError: key, value = 'meta', [[mk, mv]]
                except IndexError:
                    terminal(f'Metadata requires key and value: {arg} {mk} {mv}')
                    sys.exit(1)

            # Metadata Key
            elif key in ('meta-key', 'mk'):
                try: key, value = 'meta-key', options['meta-key']+[args.pop(0)]
                except KeyError: key, value = 'meta-key', [args.pop(0)]
                except: value = None

            # JSON metadata
            elif key in ('jm', 'jsonmeta'):
                key, value = 'jsonmeta', args.pop(0)

            # Little Endian Toggle
            elif key in ('le', 'little-endian'):
                key, value = 'le', True

            # FrAD Profile
            elif key in ('prf', 'profile'):
                prf = '<null>'
                try:
                    prf = args.pop(0)
                    key, value = 'profile', int(prf)
                except:
                    terminal(f'Value cannot be parsed as Integer: {arg} {prf}')
                    sys.exit(1)

            # Compression level
            elif key in ('lv', 'loss-level', 'level'):
                lv = '<null>'
                try:
                    lv = args.pop(0)
                    key, value = 'loss-level', int(lv)
                except:
                    terminal(f'Value cannot be parsed as Integer: {arg} {lv}')
                    sys.exit(1)

            # Verbose CLI Toggle
            elif key in ('v', 'verbose'):
                key, value = 'verbose', True

            elif key in ('ffmpeg', 'ff', 'directcmd', 'direct-cmd', 'direct-ffmpeg'):
                key, value = 'directcmd', args
                args = []

            else: value = True

            options[key] = value

    return action, file_path, metaoption, options
