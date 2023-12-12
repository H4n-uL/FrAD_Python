import argparse
import base64
import json
from fra import encode
from fra import decode
from fra import header
from fra import player
from fra import repack

def main(action, args):
    input = args.input
    output = args.output if args.output else 'fourierAnalogue.fra'
    meta = args.meta
    if args.jsonmeta is not None:
        with open(args.jsonmeta, 'r') as f:
            jsonmeta = json.load(f)
        meta = []
        # JSON의 각 키-값 쌍에 대해
        for item in jsonmeta:
            value = item["value"]
            # type이 "base64"인 경우 value를 바이트로 변환
            if item["type"] == "base64":
                value = base64.b64decode(value)
            # [key, value]를 결과 리스트에 추가
            meta.append([item["key"], value])
    img = None

    if args.image:
        with open(args.image, 'rb') as i: 
            img = i.read()

    if action == 'encode':
        encode.enc(input, int(args.bits), out=output, apply_ecc=args.ecc, new_sample_rate=int(args.nsr) if args.nsr is not None else None, meta=meta, img=img)
    elif action == 'decode':
        decode.dec(input, out=output, bits=int(args.bits), codec=args.codec, quality=args.quality)
    elif action == 'parse':
        output = args.output if args.output is not None else 'metadata'
        head, img = header.parse(input)
        result_list = []
        for key, value in head.items():
            item_dict = {"key": key}
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8')
                    item_dict["type"] = "string"
                    item_dict["value"] = value
                except UnicodeDecodeError:
                    item_dict["type"] = "base64"
                    item_dict["value"] = base64.b64encode(value).decode('utf-8')
            elif isinstance(value, int):
                item_dict["type"] = "integer"
                item_dict["value"] = str(value)
            elif isinstance(value, float):
                item_dict["type"] = "float"
                item_dict["value"] = str(value)
            result_list.append(item_dict)
        with open(output+'.meta.json', 'w') as m: m.write(json.dumps(result_list, ensure_ascii=False))
        with open(output+'.meta.image', 'wb') as m: m.write(img)
    elif action == 'modify' or action == 'meta-modify':
        print(meta)
        header.modify(input, meta=meta, img=img)
    elif action == 'ecc':
        repack.ecc(input)
    elif action == 'play':
        player.play(input, keys=int(args.keys) if args.keys is not None else None, speed_in_times=int(args.speed) if args.speed is not None else None)
    else:
        raise ValueError("Invalid action. Please choose 'encode' or 'decode'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fourier Analogue Codec')
    parser.add_argument('action', choices=['encode', 'decode', 'parse', 'modify', 'meta-modify', 'ecc', 'play'], help='action to perform')
    parser.add_argument('-i', '--input', required=True, help='input file path')
    parser.add_argument('-o', '--output', required=False, help='output file path')
    parser.add_argument('-b', '--bits', required=False, help='output file bit depth')
    parser.add_argument('-n', '--nsr', required=False, help='resample as new sample rate')
    parser.add_argument('-img', '--image', required=False, help='image file path')
    parser.add_argument('-c', '--codec', required=False, help='codec type')
    parser.add_argument('-e', '--ecc', action='store_true', help='enable ecc')
    parser.add_argument('-s', '--speed', required=False, help='play speed(in times)')
    parser.add_argument('-q', '--quality', required=False, help='decode quality')
    parser.add_argument('-k', '--keys', required=False, help='keys')
    parser.add_argument('-m', '--meta', action='append', nargs=2, required=False, help='metadata in "key" "value" format')
    parser.add_argument('-jm', '--jsonmeta', required=False, help='metadata in json')
    
    args = parser.parse_args()
    main(args.action, args)
