import numpy as np

# Modified Opus Subbands
MOS =  (0,     200,   400,   600,   800,   1000,  1200,  1400,
        1600,  2000,  2400,  2800,  3200,  4000,  4800,  5600,
        6800,  8000,  9600,  12000, 15600, 20000, 24000, 28800,
        34400, 40800, 48000, -1)

subbands = len(MOS) - 1
rndint = lambda x: int(x+0.5)

class pns:
    @staticmethod
    def getbinrng(dlen: int, smprate: int, subband_index: int) -> slice:
        return slice(rndint(dlen/(smprate/2)*MOS[subband_index]),
            None if MOS[subband_index+1] is -1 else rndint(dlen/(smprate/2)*MOS[subband_index+1]))

    @staticmethod
    def mask_thres_MOS(freqs: np.ndarray, alpha: float, smprate: int) -> np.ndarray:
        thres = np.zeros_like(freqs)
        for i in range(subbands):
            subfreqs = freqs[pns.getbinrng(len(freqs), smprate, i)]
            if len(subfreqs) > 0:
                if MOS[i+1] == -1: thres[pns.getbinrng(len(freqs), smprate, i)] = np.inf; continue
                f = (MOS[i] + MOS[i+1]) / 2
                ABS = (3.64*(f/1000.)**-0.8 - 6.5*np.exp(-0.6*(f/1000.-3.3)**2.) + 1e-3*((f/1000.)**4.))
                ABS = np.clip(ABS, None, 96)
                thres[pns.getbinrng(len(freqs), smprate, i)] = np.maximum(np.max(subfreqs)**alpha, 10.0**((ABS-96)/20))
        return thres

    @staticmethod
    def mapping2opus(freqs: np.ndarray, smprate):
        mapped_freqs = np.zeros(subbands)
        for i in range(subbands):
            subfreqs = freqs[pns.getbinrng(len(freqs), smprate, i)]
            if len(subfreqs) > 0:
                mapped_freqs[i] = np.sqrt(np.mean(subfreqs**2))
        return mapped_freqs

    @staticmethod
    def mappingfromopus(mapped_freqs, freqs_shape, smprate):
        freqs = np.zeros(freqs_shape)
        for i in range(subbands):
            freqs[pns.getbinrng(freqs_shape, smprate, i)] = mapped_freqs[i]
        return freqs

def quant(freqs: np.ndarray, channels: int, dlen: int, kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
    alpha = 0.8

    if kwargs['level'] < 11: const_factor = (kwargs['level']+1)*7/80
    else: const_factor = (kwargs['level']+1 - 10) / 2

    # Perceptual Noise Substitution
    pns_sgnl = []
    mask = []
    for c in range(channels):
        thres = pns.mask_thres_MOS(
            pns.mapping2opus(np.abs(freqs[c]),kwargs['smprate']),alpha,kwargs['smprate']) * const_factor
        mask.append(thres)
        pns_sgnl.append(np.around(freqs[c] / pns.mappingfromopus(thres,dlen,kwargs['smprate'])))

    return np.array(pns_sgnl), np.array(mask)

def dequant(pns_sgnl: np.ndarray, channels: int, masks: np.ndarray, kwargs: dict) -> np.ndarray:
    masks = np.where(np.isnan(masks) | np.isinf(masks), 0, masks)
    return np.array([pns_sgnl[c] * pns.mappingfromopus(masks[c], len(pns_sgnl[c]), kwargs['smprate']) for c in range(channels)])
