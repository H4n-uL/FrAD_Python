import base64, json, os, sys, traceback

encode_opt = ['encode']
decode_opt = ['decode']
parse_opt = ['parse']
meta_modify_opt = ['modify', 'meta-modify']
repack_ecc_opt = ['ecc', 'repack']
play_opt = ['play']
record_opt = ['record']
update_opt = ['update']

encode_help =      '''----------------------------------description-----------------------------------

Encode
This action will encode your audio file to FrAD, Preserving all metadata, image,
and original audio file.

-------------------------------------usage--------------------------------------

fourier encode path/to/audio.file --bits [bit depth] {kwargs...}

------------------------------------options-------------------------------------

    --bits            | Bit depth, REQUIRED (alias: b, bit)
                      |
    --ecc             | Enable ECC, recommended (alias: e, apply-ecc,
                      |                                              enable-ecc)
    --data-ecc        | ECC size ratio in [data size] [ecc code size],
                      |  default: 128, 20 (alias: ds, ecc-ratio, data-ecc-ratio)
                      |
    --output          | Output file path (alias: o, out, output-file)
    --sample-rate     | New sample rate (alias: sr, srate, nsr, new-srate,
                      |                               new-sample-rate, resample)
    --fsize           | Samples per frame, default: 2048 (alias: fr, frame-size,
                      |                                       samples-per-frame)
    --gain            | Gain level in both dBFS and amplitude (alias: g, gain)
    --le              | Little Endian Toggle (alias: le, little-endian)
                      |
    --meta            | Metadata in [key] [value], default: pre-embedded meta
                      |                                         (alias: m, meta)
    --jsonmeta        | Metadata in JSON format (alias: jm)
    --image           | Image to embed, default: pre-embedded image (alias: img)
                      |
    --profile         | FrAD Profile from 0 to 7, NOT RECOMMENDED (alias: prf)
    --loss-level      | Lossy compression level, default: 0 (alias: lv, level)
                      |
    --verbose         | Verbose output (alias: v)'''
decode_help =      '''----------------------------------description-----------------------------------

Decode
This action will encode any supporting FrAD files to another format. It highly
leans on ffmpeg for re-encoding.

-------------------------------------usage--------------------------------------

fourier encode path/to/audio.file {kwargs...} {--ffmpeg {ffmpeg decode command}}

------------------------------------options-------------------------------------

    --ecc             | Check errors and fix, recommended (alias: e, apply-ecc,
                      |                                              enable-ecc)
    --gain            | Gain level in both dBFS and amplitude (alias: g)
    --verbose         | Verbose output (alias: v)
                      |
    --ffmpeg          | Pass a custom FFmpeg command for decoding.
                      | recommended for advanced users. Any options specified
                      |         after --ffmpeg will be passed directly to FFmpeg.
                      |            Output file name auto-detection not supported.
                      |         (alias: ff, directcmd, direct-cmd, direct-ffmpeg)
                      |
    --codec           | Codec for decoding, default: 24-bit FLAC (alias: c)
    --quality         | Quality for decoding in [bitrate]{c|v|a},
                      |                      default: maximum quality (alias: q)
    --output          | Output file path (alias: o, out, output-file)
    --bits            | Bit depth (alias: b, bit)
    --sample-rate     | New sample rate (alias: sr, srate, nsr, new-srate,
                      |                               new-sample-rate, resample)'''
play_help =        '''----------------------------------description-----------------------------------

Play
This action will play FrAD files, not decoding to any other format.

------------------------------------options-------------------------------------

    --gain            | Gain level in both dBFS and amplitude (alias: g)
    --keys / --speed  | Keys for playback (alias: k, key) | Playback speed
                      |                                             (alias: spd)
    --ecc             | Check errors and fix while playback (alias: e,
                      |                                   apply-ecc, enable-ecc)
    --verbose         | Verbose output (alias: v)'''
record_help =      '''----------------------------------description-----------------------------------

Record
This action will capture audio stream and write directly to FrAD file.

------------------------------------options-------------------------------------

    --bits            | Bit depth, default: 24 (alias: b, bit)
    --sample-rate     | Record srate, default: 48000 (alias: sr, srate, ...)
                      |
    --ecc             | Enable ECC (alias: e, apply-ecc, enable-ecc)
    --data-ecc        | ECC size ratio in [data size] [ecc code size]
                      |  default: 128, 20 (alias: ds, ecc-ratio, data-ecc-ratio)
                      |
    --le              | Little Endian Toggle (alias: le, little-endian)
                      |
    --profile         | FrAD Profile from 0 to 7, NOT RECOMMENDED (alias: prf)
    --loss-level      | Lossy compression level (alias: lv, level)'''
repack_ecc_help =  '''----------------------------------description-----------------------------------

Repack
This action will protect FrAD files via Reed-Solomon algorithm or check and fix
errors.

------------------------------------options-------------------------------------

    --data-ecc        | ECC size ratio in [data size] [ecc code size]
                      |  default: 128, 20 (alias: ds, ecc-ratio, data-ecc-ratio)
    --verbose         | Verbose output (alias: v)'''
parse_help =       '''----------------------------------description-----------------------------------

Parse
This action will parse metadata into JSON and extract embedded image.

------------------------------------options-------------------------------------

    --output          | Output file path (alias: o, out, output-file)'''
meta_modify_help = '''----------------------------------description-----------------------------------

Modify
This action will overwrite any metadata.
WARNING: This option will delete all metadata if no option provided.
It is HIGHLY recommended to preserve your metadata via `fourier parse` before
running this action.

------------------------------------options-------------------------------------

    --meta            | Metadata in [key] [value] (alias: m, meta)
    --jsonmeta        | Metadata in JSON format (alias: jm)
    --image           | Image to embed (alias: img)'''
update_help = '''----------------------------------description-----------------------------------

Update
This action will update Fourier Analogue-in-Digital from the repository.

------------------------------------options-------------------------------------

    No option for this action.'''

def main(action, file_path, kwargs: dict):
    output = kwargs.get('output', None)
    verbose = kwargs.get('verbose', False)
    srate = kwargs.get('srate', None)
    ecc_enabled = kwargs.get('ecc', False)
    data_ecc = kwargs.get('data-ecc', [128, 20])
    loss_level = kwargs.get('loss-level', 0)
    le = kwargs.get('le', False)
    gain = kwargs.get('gain', 1)

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

    if action in encode_opt:
        from FrAD import encode
        if kwargs.get('bits', None) is None:
            print('bit depth is required for encoding.')
            sys.exit(1)
        encode.enc(
                file_path, int(kwargs['bits']), le,
                output, profile, loss_level,
                kwargs.get('fsize', 2048), gain,
                ecc_enabled, data_ecc,
                srate, meta, img, verbose)

    elif action in decode_opt:
        from FrAD import decode
        bits = kwargs.get('bits', 32)
        decode.dec(
                file_path, kwargs.get('directcmd', None), output, bits,
                kwargs.get('codec', 'flac'),
                kwargs.get('quality', None),
                ecc_enabled, gain, srate,
                verbose)

    elif action in parse_opt:
        from FrAD import header
        header.parse_file(file_path, kwargs.get('output', 'metadata'))

    elif action in meta_modify_opt:
        from FrAD import header
        header.modify(file_path, meta=meta, img=img)

    elif action in repack_ecc_opt:
        from FrAD import repack
        repack.ecc(file_path, data_ecc, kwargs['verbose'])

    elif action in play_opt:
        from FrAD import player
        player.play(
                file_path, gain, kwargs.get('keys', None),
                kwargs.get('speed', None),
                ecc_enabled, verbose)

    elif action in record_opt:
        from FrAD import recorder
        bits = kwargs.get('bits', 24)
        recorder.record_audio(file_path, kwargs.get('srate', 48000), None, bits,
            kwargs.get('fsize', 2048),
            ecc_enabled, data_ecc,
            profile, loss_level, le)

    elif action in update_opt:
        from FrAD.tools import update
        update.fetch_git('https://api.github.com/repos/h4n-ul/Fourier_Analogue-in-Digital/contents/src', os.path.dirname(__file__))
    
    elif action in ['help']:
        print(
'''               Fourier Analogue-in-Digital Master encoder/decoder
                             Original Author - HaמuL
''')
        if file_path in encode_opt:
            print(encode_help)
        elif file_path in decode_opt:
            print(decode_help)
        elif file_path in parse_opt:
            print(parse_help)
        elif file_path in meta_modify_opt:
            print(meta_modify_help)
        elif file_path in repack_ecc_opt:
            print(repack_ecc_help)
        elif file_path in play_opt:
            print(play_help)
        elif file_path in record_opt:
            print(record_help)
        elif file_path in update_opt:
            print(update_help)
        else:
            print(
'''--------------------------------available actions-------------------------------

    encode      | Encode any audio formats to FrAD
    decode      | Encode FrAD to any audio formats
    play        | Direct FrAD playback
    record      | Direct Software FrAD recording
    parse       | Parse metadata from FrAD
    ecc         | Enable/Repack ECC protection (alias: repack)
    meta-modify | Overwrite all metadata of FrAD (alias: modify)
    update      | Update FrAD codec from Github''')
        print()
    else:
        raise ValueError('Invalid action. type `fourier help` to get help.')

if __name__ == '__main__':
    from FrAD.tools.argparse import parse_args
    try:
        if len(sys.argv) == 1:
            print('Fourier Analogue-in-Digital Master encoder/decoder')
            print('Please type `fourier help` to get help.')
            sys.exit(0)
        action, file_path, kwargs = parse_args(sys.argv[1:])
        main(action, file_path, kwargs)
    except Exception as e:
        if type(e) == KeyboardInterrupt:
            sys.exit(0)
        else:
            print(traceback.format_exc())
            sys.exit(1)
