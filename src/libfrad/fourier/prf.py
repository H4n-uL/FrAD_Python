class profiles:
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

    SAMPLES_LI = tuple([item for sublist in SAMPLES.values() for item in sublist])
    MAX_SMPL = max(SAMPLES_LI)
