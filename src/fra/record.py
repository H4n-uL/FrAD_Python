from .fourier import fourier
import os, struct, sys, zlib
import sounddevice as sd
from .tools.ecc import ecc
from .tools.headb import headb
from .tools.lossy_psycho import PsychoacousticModel

class recorder:
    def record_audio(file_path, sample_rate = 48000, channels = 1,
            bit_depth = 24,
            samples_per_frame: int = 2048,
            apply_ecc: bool = False, ecc_sizes: list = ['128', '20'],
            lossy = False, loss_level: int = 0, little_endian = False,
            meta = None, img: bytes = None):

        segmax = ((2**31-1) // (((ecc_dsize+ecc_codesize)/ecc_dsize if apply_ecc else 1) * channels * 16)//16)*2
        if samples_per_frame > segmax: raise ValueError(f'Sample size cannot exceed {segmax}.')
        if samples_per_frame < 2: raise ValueError(f'Sample size must be at least 2.')
        if samples_per_frame % 2 != 0: raise ValueError('Sample size must be multiple of 2.')

        if not 20 >= loss_level >= 0: raise ValueError(f'Lossy compression level should be between 0 and 20.')

        hw = int(input(f'Please enter your recording device ID from below.\n{sd.query_devices()}\n> '))
        if lossy and 'y' not in input('\033[1m!!!Warning!!!\033[0m\nFourier Analogue-in-Digital is designed to be an uncompressed archival codec. Compression increases the difficulty of decoding and makes data very fragile, making any minor damage likely to destroy the entire frame. Proceed? (Y/N) ').lower(): sys.exit('Aborted.')

        # Setting file extension
        if not (file_path.lower().endswith('.frad') or file_path.lower().endswith('.dsin') or file_path.lower().endswith('.fra') or file_path.lower().endswith('.dsn')):
            if len(file_path) <= 8 and all(ord(c) < 128 for c in file_path): file_path += '.fra'
            else: file_path += '.frad'
        
        if os.path.exists(file_path) and 'y' not in input(f'{file_path} Already exists. Proceed? ').lower(): sys.exit('Aborted.')
        ecc_dsize, ecc_codesize = int(ecc_sizes[0]), int(ecc_sizes[1])
        print("Recording...")
        open(file_path, 'wb').write(headb.uilder(meta, img))
        with sd.InputStream(samplerate=sample_rate, channels=channels, device=hw) as record, open(file_path, 'ab') as f:
            psycho = PsychoacousticModel()
            while True:
                try:
                    data = record.read(samples_per_frame)[0]
                    flen = len(data)
                    data, bits = fourier.analogue(data, bit_depth, channels, little_endian, lossy=lossy, sample_rate=sample_rate, level=loss_level, model=psycho)
                    if lossy: data = zlib.compress(data, level=9)

                    # Applying ECC (This will make encoding hundreds of times slower)
                    if apply_ecc: data = ecc.encode(data, ecc_dsize, ecc_codesize)
                    
                    data = bytes(
                        #-- 0x00 ~ 0x0f --#
                            # Frame Signature
                            b'\xff\xd0\xd2\x97' +

                            # Segment length(Processed)
                            struct.pack('>I', len(data)) +

                            headb.encode_efb(lossy, apply_ecc, little_endian, bits) + # EFB
                            struct.pack('>B', channels - 1) +                      # Channels
                            struct.pack('>B', ecc_dsize if apply_ecc else 0) +     # ECC DSize
                            struct.pack('>B', ecc_codesize if apply_ecc else 0) +  # ECC code size

                            struct.pack('>I', sample_rate) +                       # Sample Rate

                        #-- 0x10 ~ 0x1f --#
                            b'\x00'*8 +

                            # Samples in a frame per channel
                            struct.pack('>I', flen) +

                            # ISO 3309 CRC32
                            struct.pack('>I', zlib.crc32(data)) +

                        #-- Data --#
                        data
                    )

                    # WRITE
                    f.write(data)

                except KeyboardInterrupt:
                    break

        print("Recording stopped.")
