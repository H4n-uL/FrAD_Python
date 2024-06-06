import base64, json, os, sys, traceback

def terminal(*args: object, sep: str | None = ' ', end: str | None = '\n'):
    sys.stderr.buffer.write(f'{(sep or '').join(map(str,args))}{end}'.encode())
    sys.stderr.buffer.flush()

encode_help = f'''--------------------------------- Description ----------------------------------

Encode
This action will encode your audio file to FrAD, Preserving all metadata, image,
and original audio file.

------------------------------------ Usage -------------------------------------

fourier encode path/to/audio.file --bits [bit depth] {{kwargs...}}

----------------------------------- Options ------------------------------------

    --bits        | Bit depth, REQUIRED (alias: b, bit)
                  |
    --ecc         | Enable ECC, recommended.
                  | ECC size ratio in --ecc [data size] [ecc code size]
                  | default: 96, 24 (alias: e, apply-ecc, enable-ecc)
                  |
    --output      | Output file path (alias: o, out, output-file)
    --sample-rate | Sample rate (alias: sr, srate)
    --channels    | Channels (alias: c, chnl, channel)
    --raw         | Raw PCM data flag with data type (alias: r, pcm)
                  |
    --fsize       | Samples per frame, default: 2048
                  |                   (alias: fr, frame-size, samples-per-frame)
    --gain        | Gain level in both dBFS and amplitude (alias: g, gain)
    --le          | Little Endian Toggle (alias: le, little-endian)
    --overlap     | Overlap ratio in 1/{{value}} (alias: olap)
                  |
    --meta        | Metadata in [key] [value], default: pre-embedded meta
                  |                                                   (alias: m)
    --jsonmeta    | Metadata in JSON format (alias: jm)
    --image       | Image to embed, default: pre-embedded image (alias: img)
                  |
    --profile     | FrAD Profile from 0 to 7, NOT RECOMMENDED (alias: prf)
    --loss-level  | Lossy compression level, default: 0 (alias: lv, level)
                  |
    --verbose     | Verbose output (alias: v)'''
decode_help = f'''--------------------------------- Description ----------------------------------

Decode
This action will encode any supporting FrAD files to another format. It highly
leans on ffmpeg for re-encoding.

------------------------------------ Usage -------------------------------------

fourier encode path/to/audio.file {{kwargs...}} {{--ffmpeg {{ffmpeg decode command}}}}

----------------------------------- Options ------------------------------------

    --ecc         | Check errors and fix, recommended.
                  |                            (alias: e, apply-ecc, enable-ecc)
    --gain        | Gain level in both dBFS and amplitude (alias: g)
    --verbose     | Verbose output (alias: v)
                  |
    --ffmpeg      | Pass a custom FFmpeg command for decoding.
                  | recommended for advanced users. Any options specified after
                  | --ffmpeg will be passed directly to FFmpeg.
                  | Output file name auto-detection not supported.
                  |         (alias: ff, directcmd, direct-cmd, direct-ffmpeg)
                  |
    --codec       | Codec for decoding, default: 24-bit FLAC
    --quality     | Quality for decoding in [bitrate]{{c|v|a}},
                  |                      default: maximum quality (alias: q)
    --output      | Output file path (alias: o, out, output-file)
    --bits        | Bit depth (alias: b, bit)
    --sample-rate | New sample rate (alias: sr, srate)'''
play_help = f'''--------------------------------- Description ----------------------------------

Play
This action will play FrAD files, not decoding to any other format.

----------------------------------- Options ------------------------------------

    --gain        | Gain level in both dBFS and amplitude (alias: g)
    --keys        | Keys for playback, exclusive with --speed (alias: k, key)
    --speed       | Playback speed, exclusive with --keys (alias: spd)
    --ecc         | Check errors and fix while playback
                  |                            (alias: e, apply-ecc, enable-ecc)
    --verbose     | Verbose output (alias: v)'''
record_help = f'''--------------------------------- Description ----------------------------------

Record
This action will capture audio stream and write directly to FrAD file.

----------------------------------- Options ------------------------------------

    --bits        | Bit depth, default: 24 (alias: b, bit)
    --sample-rate | Record srate, default: 48000 (alias: sr, srate)
                  |
    --ecc         | Enable ECC, NOT recommended for high bit depth and srate.
                  | ECC size ratio in --ecc [data size] [ecc code size]
                  | default: 96, 24 (alias: e, apply-ecc, enable-ecc)
                  |
    --le          | Little Endian Toggle (alias: le, little-endian)
    --overlap     | Overlap ratio in 1/{{value}} (alias: olap)
                  |
    --profile     | FrAD Profile from 0 to 7, NOT RECOMMENDED (alias: prf)
    --loss-level  | Lossy compression level (alias: lv, level)'''
repack_ecc_help = f'''--------------------------------- Description ----------------------------------

Repack
This action will protect FrAD files via Reed-Solomon algorithm or check and fix
errors.

----------------------------------- Options ------------------------------------

    --ecc         | ECC size ratio in --ecc [data size] [ecc code size]
                  | default: 96, 24 (alias: e, apply-ecc, enable-ecc)
    --verbose     | Verbose output (alias: v)'''
meta_help = f'''--------------------------------- Description ----------------------------------

Edit Metadata
This action will do actions on metadata.

------------------------------------ Usage -------------------------------------

fourier meta {{action}} path/to/audio.file {{kwargs...}}

----------------------------------- Options ------------------------------------

    add
    --meta        | Metadata in [key] [value] (alias: m, meta)
    --jsonmeta    | Metadata in JSON format (alias: jm)

    rm
    --meta-key    | Metadata key (alias: mk)

    write-img
    --image       | Image to embed (alias: img)

    rm-img
    No option for this action.

    overwrite
    --meta        | Metadata in [key] [value] (alias: m, meta)
    --jsonmeta    | Metadata in JSON format (alias: jm)
    --image       | Image to embed (alias: img)

    parse
    --output      | Output file path (alias: o, out, output-file)'''
update_help = f'''--------------------------------- Description ----------------------------------

Update
This action will update Fourier Analogue-in-Digital from the repository.

----------------------------------- Options ------------------------------------

    No option for this action.'''

def main(action: str, file_path: str | None, metaopt: str | None, kwargs: dict):
    from FrAD.tools.argparse import encode_opt, decode_opt, play_opt, record_opt, meta_opt, repack_ecc_opt, update_opt

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

    profile = kwargs.get('profile', 0)
    if profile > 7 or profile < 0: profile = 0
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
                raw=kwargs.get('raw', False), olap=kwargs.get('overlap', None),
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
        elif metaopt=='write-img': header.modify(file_path, img=img, write_img=True)
        elif metaopt=='rm-img': header.modify(file_path, remove_img=True)

        elif metaopt=='overwrite':
            terminal('This action will overwrite all metadata and image. if nothing provided, it will be removed. Proceed? (Y/N)')
            while True:
                x = input('> ').lower()
                if x == 'y': break
                if x == 'n': sys.exit('Aborted.')
            header.modify(file_path, meta=meta, img=img)
        elif metaopt=='parse': header.parse(file_path, kwargs.get('output', 'metadata'))
        else: terminal('Invalid meta option.'); sys.exit(1)

    elif action in repack_ecc_opt:
        if file_path is None: terminal('File path is required.'); sys.exit(1)
        from FrAD import repack
        repack.ecc(file_path, data_ecc, verbose)

    elif action in update_opt:
        from FrAD.tools import update
        update.fetch_git(os.path.dirname(__file__))

    elif action in ['help']:
        terminal(
'''               Fourier Analogue-in-Digital Master encoder/decoder
                             Original Author - HaמuL
''')
        if file_path in encode_opt: terminal(encode_help)
        elif file_path in decode_opt: terminal(decode_help)
        elif file_path in play_opt: terminal(play_help)
        elif file_path in record_opt: terminal(record_help)
        elif file_path in meta_opt: terminal(meta_help)
        elif file_path in repack_ecc_opt: terminal(repack_ecc_help)
        elif file_path in update_opt: terminal(update_help)
        else:
            terminal(
'''------------------------------- Available actions ------------------------------

    encode | Encode any audio formats to FrAD (alias: enc)
    decode | Encode FrAD to any audio formats (alias: dec)
    play   | Direct FrAD playback
    record | Direct Software FrAD recording   (alias: rec)
    repack | Enable/Repack ECC protection     (alias: ecc)
    meta   | Edit metadata on FrAD            (alias: metadata)
    update | Update FrAD codec from Github''')
        terminal()
    else:
        terminal(f'Invalid action: {{{action}}} type `fourier help` to get help.')
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
