from .tools.ecc import ecc
from ml_dtypes import bfloat16
import numpy as np
from scipy.io import wavfile
from scipy.fft import ifft
import struct

class decoder:
    def decode_mono(sample_rate, data, bits):
        data = data[:,0] * np.exp(1j * data[:,1])

        if bits == 32:
            wave = np.real(ifft(data))
            restored = np.int32(wave)
        elif bits == 16:
            wave = np.real(ifft(data) / 2**16)
            restored = np.int16(wave)
        elif bits == 8:
            wave = np.real(ifft(data) / 2**24)
            restored = np.int8(wave)

        return restored

    def decode_stereo(sample_rate, data, bits):
        left_freq_data = data[:, 0] * np.exp(1j * data[:, 1])
        right_freq_data = data[:, 2] * np.exp(1j * data[:, 3])

        if bits == 32:
            left_wave = np.int32(np.fft.ifft(left_freq_data).real)
            right_wave = np.int32(np.fft.ifft(right_freq_data).real)
        elif bits == 16:
            left_wave = np.int16(np.fft.ifft(left_freq_data / 2**16).real)
            right_wave = np.int16(np.fft.ifft(right_freq_data / 2**16).real)
        elif bits == 8:
            left_wave = np.int8(np.fft.ifft(left_freq_data / 2**24).real)
            right_wave = np.int8(np.fft.ifft(right_freq_data / 2**24).real)

        restored_stereo = np.column_stack((left_wave, right_wave))

        return restored_stereo

    def decode(file_path, out: str = None, bits: int = 32):
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
            ecc_opt = struct.unpack('<B', header[0x16:0x17])[0] >> 5

            f.seek(header_length)

            block = f.read()
            block = ecc.decode(block, ecc_opt)
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
                restored = decoder.decode_stereo(sample_rate, block_data, bits)
            if channels == 1:
                block_data = block_data.reshape(-1, 2)
                restored = decoder.decode_mono(sample_rate, block_data, bits)

            if out is not None and out[-4:-1]+out[-1] != '.wav':
                out += '.wav'
            wavfile.write(out if out is not None else'restored.wav', sample_rate, restored)
