class profiles:
    LOSSLESS = [0, 4]
    COMPACT = [1, 2]

class compact:
    srates = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000)
    samples = {128: [128 * 2**i for i in range(8)], 144: [144 * 2**i for i in range(8)], 192: [192 * 2**i for i in range(8)]}
    samples_li = tuple([item for sublist in samples.values() for item in sublist])
    max_sample = max(samples_li)