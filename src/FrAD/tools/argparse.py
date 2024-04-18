import sys

def parse_args(args: list):
    try: action = args.pop(0)
    except: return None, None, None
    file_path = None
    try: file_path = args.pop(0)
    except:
        if action not in ['update', 'help']:
            print('File path is required for the first argument.'); sys.exit(1)
    options = {}

    while args:
        arg: str = args.pop(0)
        if arg.startswith('-'):
            key = arg.lstrip('-')

            # Output file path
            if key in ['o', 'output', 'out', 'output-file']:
                key, value = 'output', args.pop(0)

            # Bit depth
            elif key in ['b', 'bits', 'bit']:
                try:
                    b = args.pop(0)
                    key, value = 'bits', int(b)
                except:
                    print(f'Value cannot be parsed as Integer: {arg} {b}')
                    sys.exit(1)

            # Image file path
            elif key in ['img', 'image']:
                key, value = 'image', args.pop(0)

            # New sample rate
            elif key in ['sr', 'srate', 'sample-rate', 'nsr', 'new-srate', 'new-sample-rate', 'resample']:
                try:
                    nsr = args.pop(0)
                    key, value = 'srate', int(nsr)
                except:
                    print(f'Value cannot be parsed as Integer: {arg} {nsr}')
                    sys.exit(1)

            # Samples per frame
            elif key in ['fr', 'fsize', 'frame-size', 'samples-per-frame']:
                try:
                    fsz = args.pop(0)
                    key, value = 'fsize', int(fsz)
                except:
                    print(f'Value cannot be parsed as Integer: {arg} {fsz}')
                    sys.exit(1)

            # Codec type
            elif key in ['c', 'codec']:
                key, value = 'codec', args.pop(0)

            # Gain
            elif key in ['g', 'gain']:
                try:
                    g = g_b = args.pop(0)
                    db = False
                    if g.lower().endswith('db'): g = g[:-2]; db = True
                    if g.lower().endswith('dbfs'): g = g[:-4]; db = True
                    if args[0].lower() in ['db', 'dbfs']: db = True; args.pop(0)
                    key, value = 'gain', float(g)
                    if db: value = 10 ** (value / 20)
                except:
                    print(f'Value cannot be parsed as Float: {arg} {g_b}')
                    sys.exit(1)

            # Enable ECC
            elif key in ['e', 'ecc', 'apply-ecc', 'enable-ecc']:
                key, value = 'ecc', True

            # Data/ECC ratio
            elif key in ['ds', 'data-ecc-size', 'data-ecc-ratio']:
                try:
                    d = e = '<null>'
                    d = args.pop(0)
                    e = args.pop(0)
                    key, value = 'data-ecc', [int(d), int(e)]
                except:
                    print(f'Value cannot be parsed as Integer: {arg} {d} {e}')
                    sys.exit(1)

            # Play speed
            elif key in ['spd', 'speed']:
                try:
                    spd = args.pop(0)
                    key, value = 'speed', float(spd)
                except:
                    print(f'Value cannot be parsed as Float: {arg} {spd}')
                    sys.exit(1)

            # Decode quality
            elif key in ['q', 'quality']:
                key, value = 'quality', args.pop(0)

            # Play keys
            elif key in ['k', 'keys', 'key']:
                try:
                    k = args.pop(0)
                    key, value = 'keys', float(k)
                except:
                    print(f'Value cannot be parsed as Float: {arg} {k}')
                    sys.exit(1)

            # Metadata
            elif key in ['m', 'meta', 'metadata']:
                try:
                    mk = mv = '<null>'
                    mk = args.pop(0)
                    mv = args.pop(0)
                    key, value = 'meta', options['meta']+[[mk, mv]]
                except KeyError:
                    key, value = 'meta', [[mk, mv]]
                except IndexError:
                    print(f'Metadata requires key and value: {arg} {mk} {mv}')
                    sys.exit(1)

            # JSON metadata
            elif key in ['jm', 'jsonmeta']:
                key, value = 'jsonmeta', args.pop(0)

            # Little Endian Toggle
            elif key in ['le', 'little-endian']:
                key, value = 'le', True

            # FrAD Profile
            elif key in ['prf', 'profile']:
                try:
                    prf = args.pop(0)
                    key, value = 'profile', int(prf)
                except:
                    print(f'Value cannot be parsed as Integer: {arg} {prf}')
                    sys.exit(1)

            # Compression level
            elif key in ['lv', 'loss-level', 'level']:
                try:
                    lv = args.pop(0)
                    key, value = 'loss-level', int(lv)
                except:
                    print(f'Value cannot be parsed as Integer: {arg} {lv}')
                    sys.exit(1)

            # Verbose CLI Toggle
            elif key in ['v', 'verbose']:
                key, value = 'verbose', True

            options[key] = value

    return action, file_path, options
