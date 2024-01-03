import argparse, base64, json, os, sys

def main(action, args):
    input = args.input
    meta = args.meta
    if args.jsonmeta is not None:
        with open(args.jsonmeta, 'r') as f:
            jsonmeta = json.load(f)
        meta = []
        for item in jsonmeta:
            value = item["value"]
            if item["type"] == "base64":
                value = base64.b64decode(value)
            meta.append([item["key"], value])
    img = None

    if args.image:
        with open(args.image, 'rb') as i:
            img = i.read()

    if action == 'encode':
        from fra import encode
        output = args.output if args.output else 'fourierAnalogue.fra'
        nsr = None if args.nsr == None else int(args.nsr)
        encode.enc(input, int(args.bits), out=output, apply_ecc=args.ecc, new_sample_rate=nsr, meta=meta, img=img)
    elif action == 'decode':
        from fra import decode
        bits = 32 if args.bits == None else int(args.bits)
        codec = args.codec if args.codec is not None else None
        decode.dec(input, out=args.output, bits=bits, codec=codec, quality=args.quality, e=args.ecc)
    elif action == 'parse':
        from fra import header
        output = args.output if args.output is not None else 'metadata'
        head, img = header.parse(input)
        result_list = []
        for item in head:
            key, value = item
            item_dict = {"key": key}
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8')
                    item_dict["type"] = "string"
                    item_dict["value"] = value
                except UnicodeDecodeError:
                    item_dict["type"] = "base64"
                    item_dict["value"] = base64.b64encode(value).decode('utf-8')
            result_list.append(item_dict)
        try:
            with open(output+'.meta.json', 'w') as m: m.write(json.dumps(result_list, ensure_ascii=False))
            with open(output+'.meta.image', 'wb') as m: m.write(img)
        except KeyboardInterrupt:
            os.remove(output+'.meta.json')
            os.remove(output+'.meta.image')
            sys.exit(0)
    elif action == 'modify' or action == 'meta-modify':
        from fra import header
        header.modify(input, meta=meta, img=img)
    elif action == 'ecc':
        from fra import repack
        repack.ecc(input)
    elif action == 'play':
        from fra import player
        player.play(input, keys=int(args.keys) if args.keys is not None else None, speed_in_times=float(args.speed) if args.speed is not None else None, e=args.ecc)
    else:
        raise ValueError("Invalid action. Please choose one of 'encode', 'decode', 'parse', 'modify', 'meta-modify', 'ecc', 'play'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fourier Analogue-in-Digital Codec')
    parser.add_argument('action', choices=['encode', 'decode', 'parse', 'modify', 'meta-modify', 'ecc', 'play'],    help='action to perform')
    parser.add_argument('input',                                                                                    help='input file path')
    parser.add_argument('-o',   '--output', '--out',                  required=False,                               help='output file path')
    parser.add_argument('-b',   '--bits',                             required=False,                               help='output file bit depth')
    parser.add_argument('-n',   '--nsr', '--new_sample_rate',         required=False,                               help='resample as new sample rate')
    parser.add_argument('-img', '--image',                            required=False,                               help='image file path')
    parser.add_argument('-c',   '--codec',                            required=False,                               help='codec type')
    parser.add_argument('-e',   '--ecc', '--apply_ecc', '--applyecc',                          action='store_true', help='enable ecc')
    parser.add_argument('-s',   '--speed',                            required=False,                               help='play speed(in times)')
    parser.add_argument('-q',   '--quality',                          required=False,                               help='decode quality')
    parser.add_argument('-k',   '--keys',                             required=False,                               help='keys')
    parser.add_argument('-m',   '--meta', '--metadata',               required=False, nargs=2, action='append',     help='metadata in "key" "value" format')
    parser.add_argument('-jm',  '--jsonmeta',                         required=False,                               help='metadata in json')

    args = parser.parse_args()
    try:
        main(args.action, args)
    except KeyboardInterrupt:
        sys.exit(0)
