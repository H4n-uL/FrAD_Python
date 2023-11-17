from .tools.ecc import ecc
from ml_dtypes import bfloat16
import numpy as np
from scipy.io import wavfile
from scipy.fft import ifft
import struct

class decode:
    def mono(sample_rate, data, bits):
        data = data[:,0] * np.exp(1j * data[:,1])

        if bits == 32: wave = np.int32(np.real(ifft(data)))
        elif bits == 16: wave = np.int16(np.real(ifft(data / 2**16)))
        elif bits == 8: wave = np.uint8(np.real(ifft(data / 2**24)))
        else: raise ValueError(f"Illegal value {bits} for bits: only 8, 16, and 32 bits are available for decoding.")

        return wave

    def stereo(sample_rate, data, bits):
        left_freq = data[:, 0] * np.exp(1j * data[:, 1])
        right_freq = data[:, 2] * np.exp(1j * data[:, 3])

        left_wave = right_wave = None

        left_wave = np.int32(np.fft.ifft(left_freq).real)
        right_wave = np.int32(np.fft.ifft(right_freq).real)
        if bits == 32:
            pass
        elif bits == 16:
            left_wave = np.int16(left_wave / 2**16)
            right_wave = np.int16(right_wave / 2**16)
        elif bits == 8:
            left_wave = np.uint8(left_wave / 2**24 + 2**7)
            right_wave = np.uint8(right_wave / 2**24 + 2**7)
        else: raise ValueError(f"Illegal value {bits} for bits: only 8, 16, and 32 bits are available for decoding.")

        return np.column_stack((left_wave, right_wave))

    def dec(file_path, out: str = None, bits: int = 32, eccless: bool = True):
        with open(file_path, 'rb') as f:
            header = f.read(256)

            signature = header[0x0:0xa]
            if signature != b'\x7e\x8b\xab\x89\xea\xc0\x9d\xa9\x68\x80':
                raise Exception('This is not Fourier Analogue file.')

            header_length = struct.unpack('<Q', header[0xa:0x12])[0]
            sample_rate = int.from_bytes(header[0x12:0x15], 'little')
            cfb = struct.unpack('<B', header[0x15:0x16])[0]
            cb = cfb >> 3
            fb = cfb & 0b111
            is_ecc_on = struct.unpack('<B', header[0x16:0x17])[0] >> 7

            f.seek(header_length)

            block = f.read()
            if is_ecc_on == 0b0:
                pass
            elif eccless:
                chunks = ecc.split_data(block, 128)
                block =  b''.join([bytes(chunk) for chunk in chunks])
            else:
                block = ecc.decode(block, is_ecc_on)
            # if b == 0b110:
            #     block_data = np.frombuffer(block, dtype=np.float512)
            # elif b == 0b101:
            #     block_data = np.frombuffer(block, dtype=np.float256)
            # elif b == 0b100:
            #     block_data = np.frombuffer(block, dtype=np.float128)
            if fb == 0b011:
                block_data = np.frombuffer(block, dtype=np.float64)
            elif fb == 0b010:
                block_data = np.frombuffer(block, dtype=np.float32)
            elif fb == 0b001:
                block_data = np.frombuffer(block, dtype=bfloat16)
            else:
                raise Exception('Illegal bits value.')

            if cb == 2:
                block_data = block_data.reshape(-1, 4)
                restored = decode.stereo(sample_rate, block_data, bits)
            elif cb == 1:
                block_data = block_data.reshape(-1, 2)
                restored = decode.mono(sample_rate, block_data, bits)
            else:
                raise Exception('Fourier Analogue only supports Mono and Stereo.')

            if out is not None and out[-4:-1]+out[-1] != '.wav':
                out += '.wav'
            wavfile.write(out if out is not None else'restored.wav', sample_rate, restored)
