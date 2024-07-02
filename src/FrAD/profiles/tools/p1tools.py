import numpy as np
import struct

# Modified Opus Subbands
MOS =  (0,     200,   400,   600,   800,   1000,  1200,  1400,
        1600,  2000,  2400,  2800,  3200,  4000,  4800,  5600,
        6800,  8000,  9600,  12000, 15600, 20000, 24000, 28800,
        34400, 40800, 48000, -1)

subbands = len(MOS) - 1
rndint = lambda x: int(x+0.5)

class pns:
    @staticmethod
    def getbinrng(dlen: int, srate: int, subband_index: int) -> slice:
        return slice(rndint(dlen/(srate/2)*MOS[subband_index]),
            MOS[subband_index+1] == -1 and None or rndint(dlen/(srate/2)*MOS[subband_index+1]))

    @staticmethod
    def mask_thres_MOS(freqs: np.ndarray, alpha: float) -> np.ndarray:
        thres = np.zeros_like(freqs)
        for i in range(subbands):
            if MOS[i+1] == -1: thres[i] = np.inf; continue
            f = (MOS[i] + MOS[i+1]) / 2
            ABS = (3.64*(f/1000.)**-0.8 - 6.5*np.exp(-0.6*(f/1000.-3.3)**2.) + 1e-3*((f/1000.)**4.))
            ABS = np.clip(ABS, None, 96)
            thres[i] = np.maximum(freqs[i]**alpha, 10.0**((ABS-96)/20))
        return thres

    @staticmethod
    def mapping2opus(freqs: np.ndarray, srate):
        mapped_freqs = np.zeros(subbands)
        for i in range(subbands):
            subfreqs = freqs[pns.getbinrng(len(freqs), srate, i)]
            if len(subfreqs) > 0:
                mapped_freqs[i] = np.sqrt(np.mean(subfreqs**2))
        return mapped_freqs

    @staticmethod
    def mappingfromopus(mapped_freqs, freqs_shape, srate):
        freqs = np.zeros(freqs_shape)
        for i in range(subbands):
            freqs[pns.getbinrng(freqs_shape, srate, i)] = mapped_freqs[i]
        return freqs

def quant(freqs: np.ndarray, channels: int, dlen: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    alpha = 0.8

    const_factor = 1.25**kwargs['level'] / 19 + 0.5

    # Perceptual Noise Substitution
    pns_sgnl = []
    mask = []
    for c in range(channels):
        thres = pns.mask_thres_MOS(
            pns.mapping2opus(np.abs(freqs[c]),kwargs['srate']),alpha) * const_factor
        mask.append(thres)
        pns_sgnl.append(np.around(freqs[c] / pns.mappingfromopus(thres,dlen,kwargs['srate'])))

    return np.array(pns_sgnl), np.array(mask)

def dequant(pns_sgnl: np.ndarray, channels: int, masks: np.ndarray, **kwargs) -> np.ndarray:
    masks = np.where(np.isnan(masks) | np.isinf(masks), 0, masks)
    return np.array([pns_sgnl[c] * pns.mappingfromopus(masks[c], len(pns_sgnl[c]), kwargs['srate']) for c in range(channels)])

bitstr2bytes = lambda bstr: bytes(int(bstr[i:i+8].ljust(8, '0'), 2) for i in range(0, len(bstr), 8))
bytes2bitstr = lambda b: ''.join(f'{byte:08b}' for byte in b)

def exp_golomb_rice_encode(data: np.ndarray):
    dmax = np.abs(data).max()
    k = dmax and int(np.ceil(np.log2(dmax))) or 0
    encoded = ''
    for n in data:
        n = (n>0) and (2*n-1) or (-2*n)
        binary_code = bin(n + 2**k)[2:]
        m = len(binary_code) - (k+1)
        encoded += ('0' * m + binary_code)

    return struct.pack('B', k) + bitstr2bytes(encoded)

def exp_golomb_rice_decode(dbytes: bytes):
    k = struct.unpack('B', dbytes[:1])[0]
    decoded = []
    data = bytes2bitstr(dbytes[1:])
    while data:
        try: m = data.index('1')
        except: break

        codeword, data = data[:(m*2)+k+1], data[(m*2)+k+1:]
        n = int(codeword, 2) - 2**k
        n = (n%2==1) and ((n+1)//2) or (-n//2)
        decoded.append(n)

    return np.array(decoded)
