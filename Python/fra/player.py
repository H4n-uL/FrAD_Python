from .decoder import decode
import sounddevice as sd
import threading
import time

class player:
    def play(file_path, keys: float = None, speed_in_times: float = None, e: bool = False):
        if keys and speed_in_times:
            raise ValueError('Keys and Speed parameter cannot be set at the same time.')
        elif keys and not speed_in_times:
            speed = 2**(keys/12)
        elif not keys and speed_in_times:
            speed = speed_in_times
        else: speed = 1
        decode.internal(file_path, 32, True, speed, e)
