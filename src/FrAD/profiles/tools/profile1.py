import numpy as np

class filter_tools:
    @staticmethod
    def f_SP_dB(maxfreq,nfilts):
        maxbark = filter_tools.hz2bark(maxfreq)
        sprfuncBarkdB = np.zeros(2*nfilts)
        sprfuncBarkdB[0:nfilts] = np.linspace(-maxbark*27,-8,nfilts)-23.5
        sprfuncBarkdB[nfilts:2*nfilts] = np.linspace(0,-maxbark*12.0,nfilts)-23.5
        return sprfuncBarkdB

    @staticmethod
    def sprfuncmat(sprfuncBarkdB, alpha, nfilts):
        sprfuncBVolt = 10.0**(sprfuncBarkdB/20.0 * alpha)
        indices = np.arange(nfilts)[:, None] + np.arange(nfilts)[::-1]
        indices = indices % nfilts
        return sprfuncBVolt[indices.ravel()].reshape(nfilts, nfilts)

    @staticmethod
    def maskingThresholdBark(mXbark,sprfuncmatrix,alpha,fs,nfilts):
        mTbark=np.dot(mXbark**alpha, sprfuncmatrix**alpha)
        mTbark=mTbark**(1.0/alpha)
        maxfreq=fs/2.0
        maxbark=filter_tools.hz2bark(maxfreq)
        step_bark = maxbark/(nfilts-1)
        barks=np.arange(0,nfilts)*step_bark
        f=filter_tools.bark2hz(barks)+1e-6
        LTQ=np.clip((3.64*(f/1000.)**-0.8 -6.5*np.exp(-0.6*(f/1000.-3.3)**2.)
            +1e-3*((f/1000.)**4.)),-20,120)
        mTbark=np.max((mTbark, 10.0**((LTQ-60)/20)),0)
        return mTbark

    @staticmethod
    def hz2bark(f): return 6. * np.arcsinh(f/600.)

    @staticmethod
    def bark2hz(Brk): return 600. * np.sinh(Brk/6.)

    @staticmethod
    def mappingbarkmat(fs, nfilts, nfft):
        maxbark = 6 * np.arcsinh(fs / 1200)
        step_bark = maxbark / (nfilts - 1)
        binbarks = 6 * np.arcsinh(np.linspace(0, fs/2, nfft//2+1) / 1200)

        W = np.zeros((nfilts, nfft), dtype=np.int8)
        bark_indices = np.round(binbarks / step_bark).astype(int)
        W[bark_indices, np.arange(nfft//2+1)] = 1

        freq_indices = [np.where(W[i, :(nfft//2+1)])[0] for i in range(nfilts)]
        W_inv = np.zeros((nfft//2+1, nfilts))
        for i, indices in enumerate(freq_indices):
            if len(indices) > 0:
                W_inv[indices, i] = 1 / np.sqrt(len(indices))
        return W, W_inv

    @staticmethod
    def mapping2bark(mX, W, nfft):
        nfreqs = nfft // 2
        return np.sqrt(np.square(mX[:nfreqs]) @ W[:, :nfreqs].T)

    @staticmethod
    def mappingfrombark(mTbark, W_inv, nfft):
        nfreqs = nfft // 2
        return mTbark @ W_inv[:, :nfreqs].T

nfilts = 32

class PsychoacousticModel:
    def __init__(self):
        self.models = {}

    def get_model(self, frame_size, alpha, sample_rate):
        key = (frame_size, alpha, sample_rate)

        # if not in cache
        if key not in self.models:
            # Creating new model table
            W, W_inv = filter_tools.mappingbarkmat(sample_rate, nfilts, frame_size*2)
            sprfuncBarkdB = filter_tools.f_SP_dB(sample_rate/2, nfilts)
            sprfuncmat = filter_tools.sprfuncmat(sprfuncBarkdB, alpha, nfilts)
            self.models[key] = {'W': W, 'W_inv': W_inv, 'sprfuncmat': sprfuncmat}

        # return
        return self.models[key]

def quant(freqs, channels, dlen, kwargs):
    alpha = 0.8
    const_factor = np.log2(kwargs['level']+1)+1

    # Getting psychoacoustic model
    M = kwargs['model'].get_model(dlen, alpha, kwargs['sample_rate'])

    # Masking threshold
    filtered = []
    mask_coeff = []
    for c in range(channels):
        mTbark = filter_tools.maskingThresholdBark(
            filter_tools.mapping2bark(np.abs(freqs[c]),M['W'],dlen*2)
            ,M['sprfuncmat'],alpha,kwargs['sample_rate'],nfilts)
        mask_coeff.append(mTbark)
        mTbark *= const_factor
        filtered.append(freqs[c] / filter_tools.mappingfrombark(mTbark,M['W_inv'],dlen*2)[:-1])

    return np.array(filtered), np.array(mask_coeff)

def dequant(filtered, channels, dlen, denoms, kwargs):
    return np.array([filtered[c] * filter_tools.mappingfrombark(
        denoms[c], filter_tools.mappingbarkmat(kwargs['sample_rate'], nfilts, dlen*2)[1], dlen*2)[:-1]
    for c in range(channels)])
