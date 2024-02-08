import argparse, base64, json, os, sys, traceback

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
        nsr = args.new_sample_rate is not None and int(args.new_sample_rate) or None
        encode.enc(
                input, int(args.bits),
                out=args.output,
                samples_per_block=int(args.sample_size),
                apply_ecc=args.ecc,
                ecc_sizes=args.data_ecc_size,
                nsr=nsr, meta=meta, img=img,
                verbose=args.verbose)
    elif action == 'decode':
        from fra import decode
        bits = 32 if args.bits == None else int(args.bits)
        codec = args.codec if args.codec is not None else None
        decode.dec(
                input,
                out=args.output, bits=bits,
                codec=codec, quality=args.quality,
                e=args.ecc, nsr=args.new_sample_rate,
                verbose=args.verbose)
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
        repack.ecc(input, args.verbose)
    elif action == 'play':
        from fra import player
        player.play(
                input,
                keys=int(args.keys) if args.keys is not None else None,
                speed_in_times=float(args.speed) if args.speed is not None else None,
                e=args.ecc, verbose=args.verbose)
    else:
        raise ValueError("Invalid action. Please choose one of 'encode', 'decode', 'parse', 'modify', 'meta-modify', 'ecc', 'play'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fourier Analogue-in-Digital Codec')
    parser.add_argument('action', choices=['encode', 'decode', 'parse', 'modify', 'meta-modify', 'ecc', 'play'],            help='Codec action')
    parser.add_argument('input',                                                                                            help='Input file path')
    parser.add_argument('-o',   '--output', '--out', '--output_file',       required=False,                                 help='Output file path')
    parser.add_argument('-b',   '--bits', '--bit',                          required=False,                                 help='Output file bit depth')
    parser.add_argument('-img', '--image',                                  required=False,                                 help='Image file path')
    parser.add_argument('-n',   '-nsr', '--new_sample_rate', '--resample',  required=False,          default='2048',        help='Resample as new sample rate')
    parser.add_argument('-smp', '--sample_size', '--samples_per_block',     required=False,                                 help='Samples per block')
    parser.add_argument('-c',   '--codec',                                  required=False,                                 help='Codec type')
    parser.add_argument('-e',   '--ecc', '--apply_ecc', '--applyecc', '--enable_ecc', '--enableecc', action='store_true',   help='Error Correction Code toggle')
    parser.add_argument('-ds',  '--data_ecc_size', '--data_ecc_ratio',      required=False, nargs=2, default=['128', '20'], help='Original data size and ECC data size(in Data size : ECC size)')
    parser.add_argument('-s',   '--speed',                                  required=False,                                 help='Play speed(in times)')
    parser.add_argument('-q',   '--quality',                                required=False,                                 help='Decode quality(for lossy codec decode only)')
    parser.add_argument('-k',   '--keys', '--key',                          required=False,                                 help='Play keys')
    parser.add_argument('-m',   '--meta', '--metadata',                     required=False, nargs=2, action='append',       help='Metadata in "key" "value" format')
    parser.add_argument('-jm',  '--jsonmeta',                               required=False,                                 help='Metadata in json, This will override --meta option.')
    parser.add_argument('-v',   '--verbose',                                                         action='store_true',   help='Verbose CLI Toggle')

    args = parser.parse_args()
    try:
        main(args.action, args)
    except Exception as e:
        if type(e) == KeyboardInterrupt:
            sys.exit(0)
        else:
            print(traceback.format_exc())
            sys.exit(1)
