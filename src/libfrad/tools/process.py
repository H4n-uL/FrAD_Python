import time

class ProcessInfo:
    def __init__(self):
        self.start_time = time.time()
        self.t_block = None
        self.total_size = 0
        self.duration = {}
        self.bitrate = {}

    def update(self, size: int, samples: int, srate: int):
        self.total_size += size
        self.duration[srate] = self.duration.get(srate, 0) + samples
        self.bitrate[srate] = self.bitrate.get(srate, 0) + size

    def get_duration(self) -> float:
        return sum([v / k for k, v in self.duration.items()])
    
    def get_bitrate(self) -> float:
        total_bits = sum(self.bitrate.values()) * 8
        total_duration = sum([v / k for k, v in self.duration.items()])
        return total_bits / total_duration if total_duration > 0 else 0
    
    def get_speed(self) -> float:
        encoding_time = time.time() - self.start_time
        total_duration = sum([v / k for k, v in self.duration.items()])
        return total_duration / encoding_time if encoding_time > 0 else 0
    
    def get_total_size(self) -> int: return self.total_size
    
    def block(self): self.t_block = time.time()
    def unblock(self):
        if self.t_block is not None:
            self.start_time += time.time() - self.t_block
            self.t_block = None
