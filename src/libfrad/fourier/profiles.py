LOSSLESS = [0, 4]
COMPACT = [1, 2]

class compact:
    SRATES = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000)
    @staticmethod
    def get_valid_srate(srate: int) -> int:
        return min([x for x in compact.SRATES if x >= srate])

    @staticmethod
    def get_srate_index(srate: int) -> int:
        return compact.SRATES.index(compact.get_valid_srate(srate))

    SAMPLES = [
          128,   160,   192,   224,
          256,   320,   384,   448,
          512,   640,   768,   896,
         1024,  1280,  1536,  1792,
         2048,  2560,  3072,  3584,
         4096,  5120,  6144,  7168,
         8192, 10240, 12288, 14336,
        16384, 20480, 24576, 28672
    ]

    @staticmethod
    def get_samples_min_ge(smpl: int) -> int:
        return min([x for x in compact.SAMPLES if x >= smpl])
    
    @staticmethod
    def get_samples_index(smpl: int) -> int:
        smpl = compact.get_samples_min_ge(smpl)
        return compact.SAMPLES.index(smpl)

    MAX_SMPL = max(SAMPLES)
