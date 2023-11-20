from .decoder import decode
import sounddevice as sd
import threading
import time

class player:
    def runtime(dur):
        start_time = time.time_ns()
        while sd.get_stream().active:
            print(f'{round((time.time_ns() - start_time) / 10**9, 3):.3f} s / {dur:.3f} s')
            time.sleep(1/18)
            print('\x1b[1A\x1b[2K', end='')

    def play(file_path):
        restored, sample_rate = decode.internal(file_path, 32)
        duration = len(restored) / sample_rate

        sd.play(restored, samplerate=sample_rate)
        thread = threading.Thread(target=player.runtime(duration))
        thread.start()

        sd.wait()
