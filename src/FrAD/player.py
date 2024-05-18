from .decoder import decode

class player:
    @staticmethod
    def play(file_path, gain, keys: float | None = None, speed: float | None = None, e: bool = False, verbose: bool = False):
        if keys and speed: print('Keys and Speed parameter cannot be set at the same time.'); return
        elif keys and not speed: speed = 2**(keys/12)
        elif not keys and speed: pass
        else: speed = 1
        decode.internal(file_path, True, speed=speed, e=e, gain=gain, ispipe=False, verbose=verbose)
