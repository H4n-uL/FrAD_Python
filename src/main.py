import argparse, base64, json, sys, traceback

def main(action, args):
    file_path = args.file_path
    meta = args.meta
    if args.jsonmeta is not None:
        with open(args.jsonmeta, 'r', encoding='utf-8') as f:
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
        from FrAD import encode
        if args.bits is None: raise ValueError('--bits option is required for encoding.')
        nsr = args.new_sample_rate is not None and int(args.new_sample_rate) or None
        encode.enc(
                file_path, int(args.bits), little_endian=args.little_endian,
                out=args.output, lossy=args.lossy, loss_level=int(args.losslevel),
                samples_per_frame=int(args.frame_size), gain=[args.gain, args.dbfs],
                apply_ecc=args.ecc,
                ecc_sizes=args.data_ecc_size,
                nsr=nsr, meta=meta, img=img,
                verbose=args.verbose)
    elif action == 'decode':
        from FrAD import decode
        bits = 32 if args.bits == None else int(args.bits)
        codec = args.codec if args.codec is not None else None
        decode.dec(
                file_path,
                out=args.output, bits=bits,
                codec=codec, quality=args.quality,
                e=args.ecc, gain=[args.gain, args.dbfs], nsr=args.new_sample_rate,
                verbose=args.verbose)
    elif action == 'parse':
        from FrAD import header
        output = args.output if args.output is not None else 'metadata'
        header.parse(file_path, output)
    elif action == 'modify' or action == 'meta-modify':
        from FrAD import header
        header.modify(file_path, meta=meta, img=img)
    elif action == 'ecc':
        from FrAD import repack
        repack.ecc(file_path, args.data_ecc_size, args.verbose)
    elif action == 'play':
        from FrAD import player
        player.play(
                file_path, gain=[args.gain, args.dbfs],
                keys=float(args.keys) if args.keys is not None else None,
                speed_in_times=float(args.speed) if args.speed is not None else None,
                e=args.ecc, verbose=args.verbose)
    elif action == 'record':
        from FrAD import recorder
        bits = 24 if args.bits == None else int(args.bits)
        recorder.record_audio(args.file_path, sample_rate=48000, channels=1,
            bit_depth=bits,
            apply_ecc=args.ecc, ecc_sizes=args.data_ecc_size,
            lossy=args.lossy, loss_level=int(args.losslevel), little_endian=args.little_endian)
    else:
        raise ValueError("Invalid action. Please choose one of 'encode', 'decode', 'parse', 'modify', 'meta-modify', 'ecc', 'play'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fourier Analogue-in-Digital Codec')
    parser.add_argument('action', choices=['encode', 'decode', 'parse', 'modify', 'meta-modify', 'ecc', 'play', 'record'],  help='Codec action')
    parser.add_argument('file_path',                                                                                        help='File path')
    parser.add_argument('-o',   '--output', '--out', '--output_file',       required=False,                                 help='Output file path')
    parser.add_argument('-b',   '--bits', '--bit',                          required=False,                                 help='Output file bit depth')
    parser.add_argument('-img', '--image',                                  required=False,                                 help='Image file path')
    parser.add_argument('-n',   '-nsr', '--new_sample_rate', '--resample',  required=False,                                 help='Resample as new sample rate')
    parser.add_argument('-fr', '--frame_size', '--samples_per_frame',       required=False,          default='2048',        help='Samples per frame')
    parser.add_argument('-c',   '--codec',                                  required=False,                                 help='Codec type')
    parser.add_argument('-g',   '--gain',                                   required=False,                                 help='Gain in X.X for relative amplitude')
    parser.add_argument('-db', '-dB', '--dbfs', '--dBFS',                                            action='store_true',   help='Converting gain as relative dB FS')
    parser.add_argument('-e',   '--ecc', '--apply_ecc', '--applyecc', '--enable_ecc', '--enableecc', action='store_true',   help='Error Correction Code toggle')
    parser.add_argument('-ds',  '--data_ecc_size', '--data_ecc_ratio',      required=False, nargs=2, default=['128', '20'], help='Original data size and ECC data size(in Data size : ECC size)')
    parser.add_argument('-s',   '--speed',                                  required=False,                                 help='Play speed(in times)')
    parser.add_argument('-q',   '--quality',                                required=False,                                 help='Decode quality(for lossy codec decode only)')
    parser.add_argument('-k',   '--keys', '--key',                          required=False,                                 help='Play keys')
    parser.add_argument('-m',   '--meta', '--metadata',                     required=False, nargs=2, action='append',       help='Metadata in "key" "value" format')
    parser.add_argument('-jm',  '--jsonmeta',                               required=False,                                 help='Metadata in json, This will override --meta option.')
    parser.add_argument('-le',  '--little_endian',                                                   action='store_true',   help='Little Endian Toggle')
    parser.add_argument('-l',   '--lossy',                                                           action='store_true',   help='Lossy compression Toggle, THIS OPTION IS HIGHLY RECOMMENDED NOT TO ENABLE.')
    parser.add_argument('-lv',  '--losslevel', '--level',                   required=False,          default='0',           help='Lossy compression level')
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
