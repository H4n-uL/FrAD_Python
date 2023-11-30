from .decoder import decode
import sounddevice as sd
import threading
import time

class player:
    def runtime(dur):
        start_time = time.time_ns()
        while sd.get_stream().active:
            print(f'{(time.time_ns() - start_time) / 10**9:.3f} s / {dur:.3f} s')
            time.sleep(1/60)
            print('\x1b[1A\x1b[2K', end='')

    def play(file_path, keys: float = None, speed_in_times: float = None):
        if keys and speed_in_times:
            raise ValueError('Keys and Speed parameter cannot be set at the same time.')
        restored, sample_rate = decode.internal(file_path, 32)
        if keys and not speed_in_times:
            sample_rate *= 2**(keys/12)
        elif not keys and speed_in_times:
            sample_rate *= speed_in_times
        duration = len(restored) / sample_rate

        sd.play(restored, samplerate=sample_rate)
        thread = threading.Thread(target=player.runtime(duration))
        thread.start()

        sd.wait()
