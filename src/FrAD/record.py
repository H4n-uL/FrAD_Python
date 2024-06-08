from .fourier import fourier
from .common import variables, terminal
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
        srate = kwargs.get('srate', 48000)
        channels = kwargs.get('chnl', None)

        # FrAD specifications
        bit_depth: int = kwargs.get('bits', 16)
        fsize: int = kwargs.get('fsize', 2048)
        little_endian: bool = kwargs.get('le', False)
        profile: int = kwargs.get('prf', 0)
        loss_level: int = kwargs.get('lv', 0)
        overlap: int = kwargs.get('olap', variables.overlap_rate)

        # ECC settings
        apply_ecc = kwargs.get('ecc', False)
        ecc_sizes = kwargs.get('ecc_sizes', [96, 24])
        ecc_dsize = ecc_sizes[0]
        ecc_codesize = ecc_sizes[1]

        # Metadata
        meta = kwargs.get('meta', None)
        img = kwargs.get('img', None)

        # segmax for Profile 0 = 4GiB / (intra-channel-sample size * channels * ECC mapping)
        # intra-channel-sample size = bit depth * 8, least 3 bytes(float s1e8m15)
        # ECC mapping = (block size / data size)
        segmax = {0: (2**32-1) // (((ecc_dsize+ecc_codesize)/ecc_dsize if apply_ecc else 1) * channels * max(bit_depth/8, 3)),
                    1: max(variables.p1.smpls_li)}
        if fsize > segmax[profile]: terminal(f'Sample size cannot exceed {segmax}.'); sys.exit(1)
        if profile == 1: fsize = min((x for x in variables.p1.smpls_li if x >= fsize), default=2048)
        if not 20 >= loss_level >= 0: terminal(f'Invalid compression level: {loss_level} Lossy compression level should be between 0 and 20.'); sys.exit()

        if profile in [1, 2]:
            srate = min(srate, 96000)
            if not srate in variables.p1.srates: srate = 48000

        if not isinstance(overlap, (int, float)): overlap = variables.overlap_rate
        elif overlap <= 0: overlap = 0
        elif overlap <= 0.5: overlap = int(1/overlap)
        elif overlap < 2: overlap = 2
        elif overlap > 255: overlap = 255
        if overlap%1!=0: overlap = int(overlap)
        terminal('Please enter your recording device ID from below.')
        for ind, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] != 0:
                terminal(f'{ind} {dev['name']}')
                terminal(f'    srate={dev['default_samplerate']}\t channels={dev['max_input_channels']}')
        while True:
            hw = int(input('> '))
            if hw in range(len(sd.query_devices())): break

        if channels is None: channels = sd.query_devices()[hw]['max_input_channels']

        # Setting file extension
        if not (file_path.lower().endswith('.frad') or file_path.lower().endswith('.dsin') or file_path.lower().endswith('.fra') or file_path.lower().endswith('.dsn')):
            if len(file_path) <= 8 and all(ord(c) < 128 for c in file_path): file_path += '.fra'
            else: file_path += '.frad'

        if os.path.exists(file_path):
            terminal(f'{file_path} Already exists. Proceed?')
            while True:
                x = input('> ').lower()
                if x == 'y': break
                if x == 'n': sys.exit('Aborted.')
        ecc_dsize, ecc_codesize = int(ecc_sizes[0]), int(ecc_sizes[1])
        terminal('Recording...')
        open(file_path, 'wb').write(headb.uilder(meta, img))
        prev = np.array([])

        record = sd.InputStream(samplerate=srate, channels=channels, device=hw, dtype=np.float32)
        record.start()
        with open(file_path, 'ab') as f:
            while True:
                try:
                    rlen = fsize
                    while rlen < len(prev): rlen += 128
                    # Overlap
                    if profile in [1, 2] and len(prev) != 0: rlen -= len(prev)
                    data = record.read(rlen)[0]
                    if overlap: data, prev = encode.overlap(data, prev, overlap, fsize=fsize, chnl=channels, profile=profile)
                    frame, _, chnl, bf = fourier.analogue(data, bit_depth, channels, little_endian, profile=profile, srate=srate, level=loss_level)

                    # Applying ECC (This will make encoding hundreds of times slower)
                    if apply_ecc: frame = ecc.encode(frame, ecc_dsize, ecc_codesize)

                    pfb = headb.encode_pfb(profile, apply_ecc, little_endian, bf)
                    encode.write_frame(f, frame, chnl, srate, pfb, (ecc_dsize, ecc_codesize), len(data), olap=overlap)

                except KeyboardInterrupt:
                    break
        record.close()
        terminal('Recording stopped.')
