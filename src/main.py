import base64, json, os, sys, traceback

def terminal(*args: object, sep: str | None = ' ', end: str | None = '\n'):
    sys.stderr.buffer.write(f'{(sep or '').join(map(str,args))}{end}'.encode())
    sys.stderr.buffer.flush()

DIR = os.path.dirname(__file__)

general_help = open(os.path.join(DIR, 'help', 'general.txt'), 'r').read()
encode_help = open(os.path.join(DIR, 'help', 'encode.txt'), 'r').read()
decode_help = open(os.path.join(DIR, 'help', 'decode.txt'), 'r').read()
play_help = open(os.path.join(DIR, 'help', 'play.txt'), 'r').read()
record_help = open(os.path.join(DIR, 'help', 'record.txt'), 'r').read()
repair_help = open(os.path.join(DIR, 'help', 'repair.txt'), 'r').read()
meta_help = open(os.path.join(DIR, 'help', 'metadata.txt'), 'r').read()
jsonmeta_help = open(os.path.join(DIR, 'help', 'jsonmeta.txt'), 'r').read()
profiles_help = open(os.path.join(DIR, 'help', 'profiles.txt'), 'r').read()
update_help = f'''------------------------------------ Update ------------------------------------

This will update Fourier Analogue-in-Digital from the repository.'''

def main(action: str, file_path: str | None, metaopt: str | None, kwargs: dict):
    from FrAD.tools.argparse import encode_opt, decode_opt, play_opt, record_opt, meta_opt, repair_opt, update_opt

    le = kwargs.get('le', False)
    fsize = kwargs.get('fsize', 2048)
    srate = kwargs.get('srate', None)

    ecc_enabled = kwargs.get('ecc', False)
    data_ecc = kwargs.get('data-ecc', [96, 24])
    if sum(data_ecc) > 255: terminal('Reed-Solomon supports up to 255 bytes for data and ecc code.'); sys.exit(1)

    gain = kwargs.get('gain', 1)

    output = kwargs.get('output', None)
    verbose = kwargs.get('verbose', False)

    meta = kwargs.get('meta', None)
    if kwargs.get('jsonmeta', None) is not None:
        with open(kwargs['jsonmeta'], 'r', encoding='utf-8') as f:
            jsonmeta = json.load(f)
        meta = []
        for item in jsonmeta:
            value = item['value']
            if item['type'] == 'base64':
                value = base64.b64decode(value)
            meta.append([item['key'], value])

    img = None
    if kwargs.get('image') is not None:
        img = open(kwargs['image'], 'rb').read()

    prf: int = kwargs.get('profile', 4)
    profile: int = (prf > 7 or prf < 0) and 4 or prf
    loss_level = kwargs.get('loss-level', 0)

    if action in encode_opt:
        if file_path is None: terminal('File path is required.'); sys.exit(1)
        if kwargs.get('bits', None) is None:
            terminal('bit depth is required for encoding.')
            sys.exit(1)
        from FrAD import encode
        encode.enc(
                file_path, kwargs['bits'], le=le,
                out=output, prf=profile, lv=loss_level,
                fsize=fsize, gain=gain, ecc=ecc_enabled, ecc_sizes=data_ecc,
                srate=srate, chnl=kwargs.get('chnl', None),
                raw=kwargs.get('raw', None), olap=kwargs.get('overlap', None),
                meta=meta, img=img, verbose=verbose)

    elif action in decode_opt:
        if file_path is None: terminal('File path is required.'); sys.exit(1)
        from FrAD import decode
        bits = kwargs.get('bits', 32)
        decode.dec(
                file_path, directcmd=kwargs.get('directcmd', None),
                out=output, bits=bits,
                codec=kwargs.get('codec', 'flac'),
                quality=kwargs.get('quality', None),
                ecc=ecc_enabled, gain=gain, srate=srate,
                verbose=verbose)

    elif action in play_opt:
        if file_path is None: terminal('File path is required.'); sys.exit(1)
        from FrAD import player
        player.play(
                file_path, gain, kwargs.get('keys', None),
                kwargs.get('speed', None),
                ecc_enabled, verbose)

    elif action in record_opt:
        if file_path is None: terminal('File path is required.'); sys.exit(1)
        from FrAD import recorder
        bits = kwargs.get('bits', 16)
        recorder.record_audio(file_path,
            srate=kwargs.get('srate', 48000),
            bits=bits, fsize=fsize, olap=kwargs.get('overlap', None),
            ecc=ecc_enabled, ecc_sizes=data_ecc,
            prf=profile, lv=loss_level, le=le)

    elif action in meta_opt:
        from FrAD import header
        if metaopt=='add': header.modify(file_path, meta=meta, img=img, add=True)
        elif metaopt=='rm': header.modify(file_path, meta=kwargs.get('meta-key', None), remove=True)
        elif metaopt=='rm-img': header.modify(file_path, remove_img=True)

        elif metaopt=='overwrite':
            terminal('This action will overwrite all metadata and image. if nothing provided, it will be removed. Proceed? (Y/N)')
            while True:
                terminal('> ', end='')
                x = input().lower()
                if x == 'y': break
                if x == 'n': sys.exit('Aborted.')
            header.modify(file_path, meta=meta, img=img)
        elif metaopt=='parse': header.parse(file_path, kwargs.get('output', 'metadata'))
        else: terminal('Invalid meta option.'); sys.exit(1)

    elif action in repair_opt:
        if file_path is None: terminal('File path is required.'); sys.exit(1)
        from FrAD import repack
        repack.ecc(file_path, data_ecc, verbose)

    elif action in update_opt:
        from FrAD.tools import update
        update.fetch_git(os.path.dirname(__file__))

    elif action in ['help']:
        print(
'''               Fourier Analogue-in-Digital Master encoder/decoder
                             Original Author - HaמuL
''')
        if   file_path in encode_opt: print(encode_help)
        elif file_path in decode_opt: print(decode_help)
        elif file_path in play_opt:   print(play_help)
        elif file_path in record_opt: print(record_help)
        elif file_path in meta_opt:   print(meta_help)
        elif file_path in repair_opt: print(repair_help)
        elif file_path in update_opt: print(update_help)
        elif file_path in update_opt: print(update_help)
        elif file_path in ['jsonmeta', 'jm']: print(jsonmeta_help)
        elif file_path in ['profiles', 'prf']: print(profiles_help)
        else:
            print(general_help)
        print()
    else:
        terminal(f'Invalid action "{action}", type `fourier help` to get help.')
        sys.exit(1)

if __name__ == '__main__':
    try:
        if len(sys.argv) == 1:
            terminal('Fourier Analogue-in-Digital Master encoder/decoder')
            terminal('Please type `fourier help` to get help.')
            sys.exit(0)
        from FrAD.tools.argparse import parse_args
        action, file_path, metaopt, kwargs = parse_args(sys.argv[1:])
        main(action, file_path, metaopt, kwargs)
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            sys.exit(0)
        else:
            terminal(traceback.format_exc())
            sys.exit(1)
