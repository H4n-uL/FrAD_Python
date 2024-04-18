import base64, json, os, sys, traceback

def main(action, file_path, kwargs: dict):
    output = kwargs.get('output', None)
    verbose = kwargs.get('verbose', False)
    new_srate = kwargs.get('new-srate', None)
    ecc_enabled = kwargs.get('ecc', False)
    data_ecc = kwargs.get('data-ecc', [128, 20])
    loss_level = kwargs.get('loss-level', 0)
    le = kwargs.get('le', False)
    gain = kwargs.get('gain', 1)


    if file_path is None and action not in ['update', 'help']:
        print('File path is required for the first argument.')
        sys.exit(1)

    meta = kwargs.get('meta', None)
    if kwargs.get('jsonmeta', None) is not None:
        with open(kwargs['jsonmeta'], 'r', encoding='utf-8') as f:
            jsonmeta = json.load(f)
        meta = []
        for item in jsonmeta:
            value = item["value"]
            if item["type"] == "base64":
                value = base64.b64decode(value)
            meta.append([item["key"], value])

    img = None
    if kwargs.get('image') is not None:
        img = open(kwargs['image'], 'rb').read()

    profile = kwargs.get('profile', 0)
    if profile > 7 or profile < 0: profile = 0

    if action == 'encode':
        from FrAD import encode
        if kwargs['bits'] is None: raise ValueError('--bits option is required for encoding.')
        new_srate = kwargs.get('new_sample_rate', None)
        encode.enc(
                file_path, int(kwargs['bits']), le,
                output, profile, loss_level,
                kwargs.get('fsize', 2048), gain,
                ecc_enabled, data_ecc,
                new_srate, meta, img, verbose)

    elif action == 'decode':
        from FrAD import decode
        bits = kwargs.get('bits', 32)
        decode.dec(
                file_path, output, bits,
                kwargs.get('codec', 'flac'),
                kwargs.get('quality', None),
                ecc_enabled, gain, new_srate,
                verbose)

    elif action == 'parse':
        from FrAD import header
        header.parse(file_path, kwargs.get('output', 'metadata'))

    elif action == 'modify' or action == 'meta-modify':
        from FrAD import header
        header.modify(file_path, meta=meta, img=img)

    elif action == 'ecc':
        from FrAD import repack
        repack.ecc(file_path, data_ecc, kwargs['verbose'])

    elif action == 'play':
        from FrAD import player
        player.play(
                file_path, gain, kwargs.get('keys', None),
                kwargs.get('speed', None),
                ecc_enabled, verbose)

    elif action == 'record':
        from FrAD import recorder
        bits = kwargs.get('bits', 24)
        recorder.record_audio(file_path, 48000, 1, bits,
            ecc_enabled, data_ecc,
            profile, loss_level, le)

    elif action == 'update':
        from FrAD.tools import update
        download_ffmpeg_portables = 'y' in input('Do you want to update ffmpeg portables? (Y/N): ').lower()
        update.fetch_git('https://api.github.com/repos/h4n-ul/Fourier_Analogue-in-Digital/contents/src', os.path.dirname(__file__), download_ffmpeg_portables)

    else:
        raise ValueError("Invalid action. Please choose one of 'encode', 'decode', 'parse', 'modify', 'meta-modify', 'ecc', 'play', 'update'.")

if __name__ == '__main__':
    from FrAD.tools.argparse import parse_args

    action, file_path, kwargs = parse_args(sys.argv[1:])
    try:
        main(action, file_path, kwargs)
    except Exception as e:
        if type(e) == KeyboardInterrupt:
            sys.exit(0)
        else:
            print(traceback.format_exc())
            sys.exit(1)
