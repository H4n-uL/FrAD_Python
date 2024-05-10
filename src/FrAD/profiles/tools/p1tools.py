import numpy as np

# Modified Opus Subbands
MOS =  [0,     200,   400,   600,   800,   1000,  1200,  1400,
        1600,  2000,  2400,  2800,  3200,  4000,  4800,  5600,
        6800,  8000,  9600,  12000, 15600, 20000, 24000, 28800,
        34400, 40800, 48000, None]

subbands = len(MOS) - 1
rndint = lambda x: int(x+0.5)

class pns:
    @staticmethod
    def getbinrng(dlen: int, smprate: int, subband_index: int) -> slice:
        return slice(rndint(dlen/(smprate/2)*MOS[subband_index]),
            None if MOS[subband_index+1] is None else rndint(dlen/(smprate/2)*MOS[subband_index+1]))

    @staticmethod
    def maskingThresholdOpus(mX: np.ndarray, alpha: float, smprate: int) -> np.ndarray:
        mT = np.zeros_like(mX)
        for i in range(subbands):
            subband_mX = mX[pns.getbinrng(len(mX), smprate, i)]
            if len(subband_mX) > 0:
                f = (MOS[i] + MOS[i+1]) / 2
                ABS = (3.64*(f/1000.)**-0.8 - 6.5*np.exp(-0.6*(f/1000.-3.3)**2.) + 1e-3*((f/1000.)**4.))
                ABS = np.clip(ABS, None, 96)
                mT[pns.getbinrng(len(mX), smprate, i)] = np.maximum(np.max(subband_mX)**alpha, 10.0**((ABS-96)/20))
        return mT

    @staticmethod
    def mapping2opus(mX: np.ndarray, smprate):
        mapped_mX = np.zeros(subbands)
        for i in range(subbands):
            subband_mX = mX[pns.getbinrng(len(mX), smprate, i)]
            if len(subband_mX) > 0:
                mapped_mX[i] = np.sqrt(np.mean(subband_mX**2))
        return mapped_mX

    @staticmethod
    def mappingfromopus(mapped_mX, mX_shape, smprate):
        mX = np.zeros(mX_shape)
        for i in range(subbands):
            mX[pns.getbinrng(mX_shape, smprate, i)] = mapped_mX[i]
        return mX

def quant(freqs: np.ndarray, channels: int, dlen: int, kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
    alpha = 0.8

    if kwargs['level'] < 11: const_factor = (kwargs['level']+1)*7/80
    else: const_factor = (kwargs['level']+1 - 10) / 2

    # Perceptual Noise Substitution
    pns_sgnl = []
    mask = []
    for c in range(channels):
        mT = pns.maskingThresholdOpus(
            pns.mapping2opus(np.abs(freqs[c]),kwargs['smprate']),alpha,kwargs['smprate']) * const_factor
        mask.append(mT)
        pns_sgnl.append(np.around(freqs[c] / pns.mappingfromopus(mT,dlen,kwargs['smprate'])))

    return np.array(pns_sgnl), np.array(mask)

def dequant(pns_sgnl: np.ndarray, channels: int, masks: np.ndarray, kwargs: dict) -> np.ndarray:
    return np.array([pns_sgnl[c] * pns.mappingfromopus(masks[c], len(pns_sgnl[c]), kwargs['smprate']) for c in range(channels)])
