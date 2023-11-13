from ml_dtypes import bfloat16
import numpy as np
import struct
import sounddevice as sd

class player:
    def play_mono(sample_rate, data):
        freq_data = data[:, 0] * np.exp(1j * data[:, 1])
        wave = np.fft.ifft(freq_data).real
        wave = wave / np.max(np.abs(wave))

        return wave

    def play_stereo(sample_rate, data):
        left_freq_data = data[:, 0] * np.exp(1j * data[:, 1])
        right_freq_data = data[:, 2] * np.exp(1j * data[:, 3])

        left_wave = np.fft.ifft(left_freq_data).real
        right_wave = np.fft.ifft(right_freq_data).real

        left_wave = left_wave / np.max(np.abs(left_wave))
        right_wave = right_wave / np.max(np.abs(right_wave))

        stereo_wave = np.stack((left_wave, right_wave), axis=1)

        return stereo_wave

    def play(file_path):
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

            f.seek(header_length)

            block = f.read()
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
                wave = player.play_stereo(sample_rate, block_data)
            if channels == 1:
                block_data = block_data.reshape(-1, 2)
                wave = player.play_mono(sample_rate, block_data)

            sd.play(wave, samplerate=sample_rate)
            sd.wait()
