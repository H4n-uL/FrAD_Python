from libfrad import common, head
try:                from .tools.cli import CliParams, META_ADD, META_OVERWRITE, META_PARSE, META_REMOVE, META_RMIMG
except ImportError: from tools.cli import CliParams, META_ADD, META_OVERWRITE, META_PARSE, META_REMOVE, META_RMIMG
import json, os, tempfile

def modify(file: str, modtype: str, params: CliParams):
    if file == '': print('Input file must be given'); exit(1)
    elif not os.path.exists(file): print('Input file does not exist'); exit(1)

    rfile = open(file, 'rb')
    he = rfile.read(64)
    head_len = 0
    if he[0:4] == common.SIGNATURE: head_len = int.from_bytes(he[8:16], 'big')
    elif he[0:4] == common.FRM_SIGN: pass
    else: print('It seems this is not a valid FrAD file.'); exit(1)

    rfile.seek(0)
    head_old = rfile.read(head_len)

    meta_old, img_old = head.parser(head_old)
    meta_new, img_new = [], b''

    if modtype == META_PARSE:
        json_ = []
        for key, data in meta_old:
            data_str = data.decode('utf-8')
            itype = 'string' if data_str.isascii() else 'base64'
            json_.append({'key': key, 'type': itype, 'value': data_str})
        open(f'{file}.json', 'w').write(json.dumps(json_, indent=2))
        if img_old: open(f'{file}.image', 'wb').write(img_old)
        return

    temp = tempfile.NamedTemporaryFile()
    temp.write(rfile.read())

    img = b''
    if os.path.exists(params.image_path):
        img = open(params.image_path, 'rb').read()

    if modtype == META_ADD:
        if meta_old: meta_new.extend(meta_old)
        meta_new.extend(params.meta)
        if img_old: img_new = img_old
        if img: img_new = img

    elif modtype == META_REMOVE:
        meta_new = [meta for meta in meta_old if meta[0] not in [meta[0] for meta in params.meta]]
        img_new = img_old

    elif modtype == META_RMIMG:
        meta_new = meta_old
        img_new = b''

    elif modtype == META_OVERWRITE:
        meta_new = params.meta
        img_new = img

    else: print('Invalid modification type.'); exit(1)

    head_new = head.builder(meta_new, img_new)

    rfile.close()
    wfile = open(file, 'wb')
    wfile.write(head_new)
    temp.seek(0)
    wfile.write(temp.read())
