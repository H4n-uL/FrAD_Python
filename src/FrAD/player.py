from .common import methods
from .decoder import decode

class player:
    def play(file_path, gain, keys: float = None, speed_in_times: float = None, e: bool = False, verbose: bool = False):
        if keys and speed_in_times:
            raise ValueError('Keys and Speed parameter cannot be set at the same time.')
        elif keys and not speed_in_times:
            speed = 2**(keys/12)
        elif not keys and speed_in_times:
            speed = speed_in_times
        else: speed = 1
        decode.internal(file_path, True, speed, e, gain, verbose)
