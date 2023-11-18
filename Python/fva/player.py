from .decoder import decode
import sounddevice as sd

class player:
    def play(file_path):
        restored, sample_rate = decode.internal(file_path, 32)
        sd.play(restored, samplerate=sample_rate)
        sd.wait()
