import numpy as np

subbands = [0,     200,   400,   600,   800,   1000,  1200,  1400,
            1600,  2000,  2400,  2800,  3200,  4000,  4800,  5600,
            6800,  8000,  9600,  12000, 15600, 20000, 24000, 28800,
            34400, 40800, 48000, None]

nfilts = len(subbands) - 1

rndint = lambda x: int(x+0.5)

class filter_tools:
    @staticmethod
    def getbinrng(dlen, sample_rate, subband_index):
        return slice(rndint(dlen/(sample_rate/2)*subbands[subband_index]),
            None if subbands[subband_index+1] is None else rndint(dlen/(sample_rate/2)*subbands[subband_index+1]))

    @staticmethod
    def maskingThresholdOpus(mX, alpha, fs):
        mT = np.zeros_like(mX)
        for i in range(nfilts):
            subband_mX = mX[filter_tools.getbinrng(len(mX), fs, i)]
            if len(subband_mX) > 0: mT[filter_tools.getbinrng(len(mX), fs, i)] = np.max(subband_mX) ** alpha
        return mT

    @staticmethod
    def mapping2opus(mX, fs):
        mapped_mX = np.zeros(nfilts)
        for i in range(nfilts):
            subband_mX = mX[filter_tools.getbinrng(len(mX), fs, i)]
            if len(subband_mX) > 0:
                mapped_mX[i] = np.sqrt(np.mean(subband_mX**2))
        return mapped_mX

    @staticmethod
    def mappingfromopus(mapped_mX, mX_shape, fs):
        mX = np.zeros(mX_shape)
        for i in range(nfilts):
            mX[filter_tools.getbinrng(mX_shape, fs, i)] = mapped_mX[i]
        return mX

class PsychoacousticModel:
    def __init__(self):
        self.models = {}

def quant(freqs, channels, dlen, kwargs):
    alpha = 0.8
    const_factor = 0.25 + kwargs['level']/16

    # Masking threshold
    filtered = []
    mask = []
    for c in range(channels):
        mT = filter_tools.maskingThresholdOpus(
            filter_tools.mapping2opus(np.abs(freqs[c]),kwargs['sample_rate']),alpha,kwargs['sample_rate'])
        mask.append(mT)
        mT *= const_factor
        mT = filter_tools.mappingfromopus(mT,dlen,kwargs['sample_rate'])
        filtered.append(freqs[c] * np.where(mT > 0, 1, 0) / np.where(mT > 0, mT, 1))

    return np.array(filtered), np.array(mask)

def dequant(filtered, channels, masks, kwargs):
    return np.array([filtered[c] * filter_tools.mappingfromopus(masks[c], len(filtered[c]), kwargs['sample_rate']) for c in range(channels)])
