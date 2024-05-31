from .fourier import fourier
from .encoder import encode
import numpy as np
import math, os, sys
import sounddevice as sd
from .tools.ecc import ecc
from .tools.headb import headb

class recorder:
    @staticmethod
    def record_audio(file_path, **kwargs):
        # Audio settings
        smprate = kwargs.get('srate', 48000)
        channels = kwargs.get('chnl', None)

        # FrAD specifications
        bit_depth = kwargs.get('bits', 16)
        fsize: int = kwargs.get('fsize', 2048)
        little_endian: bool = kwargs.get('le', False)
        profile: int = kwargs.get('prf', 0)
        loss_level: int = kwargs.get('lv', 0)

        # ECC settings
        apply_ecc = kwargs.get('ecc', False)
        ecc_sizes = kwargs.get('ecc_sizes', [96, 24])
        ecc_dsize = ecc_sizes[0]
        ecc_codesize = ecc_sizes[1]

        # Metadata
        meta = kwargs.get('meta', None)
        img = kwargs.get('img', None)

        segmax = ((2**31-1) // (((ecc_dsize+ecc_codesize)/ecc_dsize if apply_ecc else 1) * 256 * 16)//16)
        if fsize > segmax: print(f'Sample size cannot exceed {segmax}.'); sys.exit()
        if fsize < 2: print(f'Sample size must be at least 2.'); sys.exit()
        if fsize % 2 != 0: print('Sample size must be multiple of 2.'); sys.exit()
        if not 20 >= loss_level >= 0: print(f'Invalid compression level: {loss_level} Lossy compression level should be between 0 and 20.'); sys.exit()
        if profile == 2 and fsize%8!=0: print(f'Invalid frame size {fsize} Frame size should be multiple of 8 for Profile 2.'); sys.exit()
        if profile in [1]:
            for mult in [128, 144, 192, None]:
                if mult == None: print(f'Invalid frame size: {fsize}, Frame size should be [128, 144, 192] * 2^({{0~7}}).'); sys.exit(1)
                if fsize == mult * (2**int(math.log2(fsize / mult))): break

        if overlap < 1: overlap = 1/overlap
        if overlap%1!=0: overlap = int(overlap)
        if overlap > 255: overlap = 255
        print('Please enter your recording device ID from below.')
        for ind, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] != 0:
                print(f'{ind} {dev['name']}')
                print(f'    srate={dev['default_samplerate']}\t channels={dev['max_input_channels']}')
        while True:
            hw = int(input('> '))
            if hw in range(len(sd.query_devices())): break

        if channels is None: channels = sd.query_devices()[hw]['max_input_channels']

        # Setting file extension
        if not (file_path.lower().endswith('.frad') or file_path.lower().endswith('.dsin') or file_path.lower().endswith('.fra') or file_path.lower().endswith('.dsn')):
            if len(file_path) <= 8 and all(ord(c) < 128 for c in file_path): file_path += '.fra'
            else: file_path += '.frad'

        if os.path.exists(file_path):
            print(f'{file_path} Already exists. Proceed?')
            while True:
                x = input('> ').lower()
                if x == 'y': break
                if x == 'n': sys.exit('Aborted.')
        ecc_dsize, ecc_codesize = int(ecc_sizes[0]), int(ecc_sizes[1])
        print('Recording...')
        open(file_path, 'wb').write(headb.uilder(meta, img))
        prev = np.array([])

        record = sd.InputStream(samplerate=smprate, channels=channels, device=hw, dtype=np.float32)
        record.start()
        with open(file_path, 'ab') as f:
            while True:
                try:
                    rlen = fsize
                    while rlen < len(prev): rlen += 128
                    # Overlap
                    if profile in [1, 2] and len(prev) != 0: rlen -= len(prev)
                    data = record.read(rlen)[0]
                    data, prev = encode.overlap(data, prev, overlap, fsize=fsize, chnl=channels, profile=profile)
                    frame, _, chnl, bf = fourier.analogue(data, bit_depth, channels, little_endian, profile=profile, smprate=smprate, level=loss_level)

                    # Applying ECC (This will make encoding hundreds of times slower)
                    if apply_ecc: frame = ecc.encode(frame, ecc_dsize, ecc_codesize)

                    pfb = headb.encode_pfb(profile, apply_ecc, little_endian, bf)
                    encode.write_frame(f, frame, chnl, smprate, pfb, (ecc_dsize, ecc_codesize), len(data))

                except KeyboardInterrupt:
                    break
        record.close()
        print('Recording stopped.')
