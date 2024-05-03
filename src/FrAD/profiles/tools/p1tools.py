import numpy as np

subbands = [0,     200,   400,   600,   800,   1000,  1200,  1400,
            1600,  2000,  2400,  2800,  3200,  4000,  4800,  5600,
            6800,  8000,  9600,  12000, 15600, 20000, 24000, 28800,
            34400, 40800, 48000, None]

nfilts = len(subbands) - 1

rndint = lambda x: int(x+0.5)

class pns:
    @staticmethod
    def getbinrng(dlen, sample_rate, subband_index):
        return slice(rndint(dlen/(sample_rate/2)*subbands[subband_index]),
            None if subbands[subband_index+1] is None else rndint(dlen/(sample_rate/2)*subbands[subband_index+1]))

    @staticmethod
    def maskingThresholdOpus(mX, alpha, fs):
        mT = np.zeros_like(mX)
        for i in range(nfilts):
            subband_mX = mX[pns.getbinrng(len(mX), fs, i)]
            if len(subband_mX) > 0:
                f = (subbands[i] + subbands[i+1]) / 2
                ABS = (3.64*(f/1000.)**-0.8 - 6.5*np.exp(-0.6*(f/1000.-3.3)**2.) + 1e-3*((f/1000.)**4.))
                ABS = np.clip(ABS, None, 96)
                mT[pns.getbinrng(len(mX), fs, i)] = np.maximum(np.max(subband_mX)**alpha, 10.0**((ABS-96)/20))
        return mT

    @staticmethod
    def mapping2opus(mX, fs):
        mapped_mX = np.zeros(nfilts)
        for i in range(nfilts):
            subband_mX = mX[pns.getbinrng(len(mX), fs, i)]
            if len(subband_mX) > 0:
                mapped_mX[i] = np.sqrt(np.mean(subband_mX**2))
        return mapped_mX

    @staticmethod
    def mappingfromopus(mapped_mX, mX_shape, fs):
        mX = np.zeros(mX_shape)
        for i in range(nfilts):
            mX[pns.getbinrng(mX_shape, fs, i)] = mapped_mX[i]
        return mX

def quant(freqs, channels, dlen, kwargs):
    alpha = 0.8

    if kwargs['level'] < 11: const_factor = (kwargs['level']+1)*7/80
    else: const_factor = (kwargs['level']+1 - 10) / 2

    # Perceptual Noise Substitution
    pns_sgnl = []
    mask = []
    for c in range(channels):
        mT = pns.maskingThresholdOpus(
            pns.mapping2opus(np.abs(freqs[c]),kwargs['sample_rate']),alpha,kwargs['sample_rate'])
        mask.append(mT)
        mT *= const_factor
        mT = pns.mappingfromopus(mT,dlen,kwargs['sample_rate'])
        pns_sgnl.append(np.around(freqs[c] / mT))

    return np.array(pns_sgnl), np.array(mask)

def dequant(pns_sgnl, channels, masks, kwargs):
    return np.array([pns_sgnl[c] * pns.mappingfromopus(masks[c], len(pns_sgnl[c]), kwargs['sample_rate']) for c in range(channels)])
