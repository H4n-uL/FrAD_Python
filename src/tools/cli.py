import base64, json

ENCODE_OPT = ['encode', 'enc']
DECODE_OPT = ['decode', 'dec']
REPAIR_OPT = ['repair', 'ecc']
PLAY_OPT = ['play', 'p']
METADATA_OPT = ['meta', 'metadata']
JSONMETA_OPT = ['jsonmeta', 'jm']
VORBISMETA_OPT = ['vorbismeta', 'vm']
PROFILES_OPT = ['profiles', 'prf']
HELP_OPT = ['help', 'h', '?']

META_ADD = 'add'
META_REMOVE = 'remove'
META_RMIMG = 'rm-img'
META_OVERWRITE = 'overwrite'
META_PARSE = 'parse'

class CliParams:
    def __init__(self):
        self.output = ''
        self.pcm = 'f64be'
        self.bits = 0
        self.srate = 0
        self.channels = 0
        self.frame_size = 2048
        self.little_endian = False
        self.profile = 4
        self.overlap_ratio = 16
        self.losslevel = 0
        self.enable_ecc = False
        self.ecc_ratio = (int(96), int(24))
        self.overwrite = False
        self.meta: list[tuple[str, bytes]] = []
        self.image_path = ''
        self.loglevel = 0
        self.speed = 1.0

    def set_meta_from_json(self, meta_path: str):
        contents = open(meta_path).read()
        json_meta = json.loads(contents)
        for item in json_meta:
            key = item['key']
            item_type = item['type']
            value_str = item['value']

            if key is None and value_str is None: continue
            key = key or ''
            value_str = value_str or ''

            value = base64.standard_b64decode(value_str) if item_type == 'base64' else value_str.encode()
            self.meta.append((key, value))

    def set_meta_from_vorbis(self, meta_path: str):
        contents = open(meta_path).readlines()

        for line in contents:
            parts = line.rstrip('\n').split('=', 1)
            if len(parts) == 1:
                if self.meta:
                    self.meta[-1] = self.meta[-1][0], self.meta[-1][1] + f'\n{parts[0]}'.encode()
                else: self.meta.append(('', parts[0].encode()))
            else: self.meta.append((parts[0], parts[1].encode()))

    def set_loglevel(self, value: str): self.loglevel = int(value)

def parse(args: list[str]):
    params = CliParams()
    executable = args.pop(0)
    if not args: return ('', '', '', params)

    action = args.pop(0).lower()
    metaaction = ''
    if action in METADATA_OPT:
        metaaction = args.pop(0).lower() if args else exit(f'Metadata action not specified, type `{executable} help meta` for available options.')
    if not args: return (action, '', '', params)
    input_file = args.pop(0)

    while args:
        key = args.pop(0).lower()

        if key.startswith('-'):
            key = key[1:]

            if key in ('output', 'out', 'o'):
                params.output = args.pop(0)
            elif key in ('pcm', 'format', 'fmt', 'f'):
                params.pcm = args.pop(0)
            elif key in ('ecc', 'enable-ecc', 'e'):
                params.enable_ecc = True
                if args and args[0].isnumeric():
                    params.ecc_ratio = int(args.pop(0)), int(args.pop(0))
            elif key in ('y', 'force'):
                params.overwrite = True
            elif key in ('bits', 'bit', 'b'):
                params.bits = int(args.pop(0))
            elif key in ('srate', 'sample-rate', 'sr'):
                params.srate = int(args.pop(0))
            elif key in ('chnl', 'channels', 'channel', 'ch'):
                params.channels = int(args.pop(0))
            elif key in ('frame-size', 'fsize', 'fr'):
                params.frame_size = int(args.pop(0))
            elif key in ('overlap-ratio', 'overlap', 'olap'):
                params.overlap_ratio = int(args.pop(0))
            elif key in ('le', 'little-endian'):
                params.little_endian = True
            elif key in ('profile', 'prf', 'p'):
                params.profile = int(args.pop(0))
            elif key in ('losslevel', 'level', 'lv'):
                params.losslevel = int(args.pop(0))
            elif key in ('tag', 'meta', 'm'):
                value = args.pop(0)
                if metaaction == META_REMOVE:
                    params.meta.append((value, b''))
                else:
                    params.meta.append((value, args.pop(0).encode()))
            elif key in ('jsonmeta', 'jm'):
                params.set_meta_from_json(args.pop(0))
            elif key in ('vorbismeta', 'vm'):
                params.set_meta_from_vorbis(args.pop(0))
            elif key in ('img', 'image'):
                params.image_path = args.pop(0)
            elif key in ('log', 'v'):
                if args and args[0].isnumeric():
                    params.set_loglevel(args.pop(0))
                else: params.set_loglevel('1')

            elif key in ('speed', 'spd'):
                params.speed = float(args.pop(0))
            elif key in ('keys', 'key', 'k'):
                params.speed = 2 ** (float(args.pop(0)) / 12)

    return (action, metaaction, input_file, params)
