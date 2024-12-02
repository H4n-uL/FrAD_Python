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

    SAMPLES = {128: [128 * 2**i for i in range(8)], 144: [144 * 2**i for i in range(8)], 192: [192 * 2**i for i in range(8)]}

    @staticmethod
    def get_samples_from_value(key: int) -> int:
        return [k for k, v in compact.SAMPLES.items() if key in v][0]

    SAMPLES_LI = [
          128,   144,   192,
          256,   288,   384,
          512,   576,   768,
         1024,  1152,  1536,
         2048,  2304,  3072,
         4096,  4608,  6144,
         8192,  9216, 12288,
        16384, 18432, 24576
    ]

    MAX_SMPL = max(SAMPLES_LI)
