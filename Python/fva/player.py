from .decoder import decode
from .tools.ecc import ecc
from ml_dtypes import bfloat16
import numpy as np
import struct
import sounddevice as sd

class player:
    def play(file_path, correct_error: bool = False):
        with open(file_path, 'rb') as f:
            header = f.read(256)

            signature = header[0x0:0xa]
            if signature != b'\x7e\x8b\xab\x89\xea\xc0\x9d\xa9\x68\x80':
                raise Exception('This is not Fourier Analogue file.')
            header_length = struct.unpack('<Q', header[0xa:0x12])[0]
            sample_rate = int.from_bytes(header[0x12:0x15], 'little')
            cfb = struct.unpack('<B', header[0x15:0x16])[0]
            channels = cfb >> 3
            bits = cfb & 0b111
            is_ecc_on = True if (struct.unpack('<B', header[0x16:0x17])[0] >> 7) == 0b1 else False

            f.seek(header_length)

            block = f.read()
            if is_ecc_on == False:
                pass
            elif correct_error != True:
                chunks = ecc.split_data(block, 128)
                block =  b''.join([bytes(chunk[:112]) for chunk in chunks])
            else:
                block = ecc.decode(block, is_ecc_on)
            # if bits == 0b110:
            #     block_data = np.frombuffer(block, dtype=np.float512)
            # elif bits == 0b101:
            #     block_data = np.frombuffer(block, dtype=np.float256)
            # elif bits == 0b100:
            #     block_data = np.frombuffer(block, dtype=np.float128)
            if bits == 0b011:
                block_data = np.frombuffer(block, dtype=np.float64)
            elif bits == 0b010:
                block_data = np.frombuffer(block, dtype=np.float32)
            elif bits == 0b001:
                block_data = np.frombuffer(block, dtype=bfloat16)
            else:
                raise Exception('Illegal bits value.')

            if channels == 2:
                block_data = block_data.reshape(-1, 4)
                wave = decode.stereo(sample_rate, block_data, 32)
            if channels == 1:
                block_data = block_data.reshape(-1, 2)
                wave = decode.mono(sample_rate, block_data, 32)

            sd.play(wave, samplerate=sample_rate)
            sd.wait()
