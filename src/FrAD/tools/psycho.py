import numpy as np

class filter_tools:
    def f_SP_dB(maxfreq,nfilts):
        maxbark = filter_tools.hz2bark(maxfreq)
        sprfuncBarkdB = np.zeros(2*nfilts)
        sprfuncBarkdB[0:nfilts] = np.linspace(-maxbark*27,-8,nfilts)-23.5
        sprfuncBarkdB[nfilts:2*nfilts] = np.linspace(0,-maxbark*12.0,nfilts)-23.5
        return sprfuncBarkdB

    def sprfuncmat(sprfuncBarkdB, alpha, nfilts):
        sprfuncBVolt = 10.0**(sprfuncBarkdB/20.0 * alpha)
        indices = np.arange(nfilts)[:, None] + np.arange(nfilts)[::-1]
        indices = indices % nfilts
        return sprfuncBVolt[indices.ravel()].reshape(nfilts, nfilts)

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

    def hz2bark(f): return 6. * np.arcsinh(f/600.)

    def bark2hz(Brk): return 600. * np.sinh(Brk/6.)

    def mapping2barkmat(fs, nfilts, nfft):
        maxbark = filter_tools.hz2bark(fs/2)
        step_bark = maxbark/(nfilts-1)
        binbark = filter_tools.hz2bark(np.linspace(0,(nfft/2),int(nfft/2)+1)*fs/nfft)
        W = np.zeros((nfilts, nfft))
        for i in range(nfilts):
            W[i,0:int(nfft/2)+1] = (np.round(binbark/step_bark)== i)
        return W

    def mapping2bark(mX,W,nfft):
        nfreqs=int(nfft/2)
        mXbark = (np.dot( np.abs(mX[:nfreqs])**2.0, W[:, :nfreqs].T))**(0.5)
        return mXbark

    def mappingfrombarkmat(W,nfft):
        nfreqs=int(nfft/2)
        W_inv= np.dot(np.diag((1.0/(np.sum(W,1)+1e-6))**0.5), W[:,0:nfreqs + 1]).T
        return W_inv

    def mappingfrombark(mTbark,W_inv,nfft):
        nfreqs=int(nfft/2)
        mT = np.dot(mTbark, W_inv[:, :nfreqs].T)
        return mT

nfilts = 64

class PsychoacousticModel:
    def __init__(self):
        self.models = {}

    def get_model(self, frame_size, alpha, sample_rate):
        key = (frame_size, alpha, sample_rate)

        # if not in cache
        if key not in self.models:
            # Creating new model table
            W = filter_tools.mapping2barkmat(sample_rate, nfilts, frame_size*2)
            W_inv = filter_tools.mappingfrombarkmat(W, frame_size*2)
            sprfuncBarkdB = filter_tools.f_SP_dB(sample_rate/2, nfilts)
            sprfuncmat = filter_tools.sprfuncmat(sprfuncBarkdB, alpha, nfilts)
            self.models[key] = {'W': W, 'W_inv': W_inv, 'sprfuncmat': sprfuncmat}

        # return
        return self.models[key]

class loss:
    def filter(freqs, channels, dlen, kwargs):
        alpha = (800 - (1.2**kwargs['level']))*0.001

        # Getting psychoacoustic model
        M = kwargs['model'].get_model(dlen, alpha, kwargs['sample_rate'])

        # Rounding off
        rounder = np.zeros(dlen)
        fl = [20, 100, 500, 2000, 5000, 10000, 20000, 100000, 500000, np.inf]
        rfs = [0, 1, 3, 4, 1, 0, -2, -3, -4]
        fs_list = {n:loss.get_range(dlen, kwargs['sample_rate'], n) for n in fl}
        for i in range(len(fl[:-1])):
            rounder[fs_list[fl[i]]:fs_list[fl[i+1]]] = 2**np.round(np.log2(dlen) - 11 - rfs[i])

        # Masking threshold
        for c in range(channels):
            # idk i just copied from open source model someone please replace it with a better one w. ERB scale
            mXbark = filter_tools.mapping2bark(np.abs(freqs[c]),M['W'],dlen*2)
            mTbark = filter_tools.maskingThresholdBark(mXbark,M['sprfuncmat'],alpha,kwargs['sample_rate'],nfilts) * np.log2(kwargs['level']*4+1)/2
            thres =  filter_tools.mappingfrombark(mTbark,M['W_inv'],dlen*2)[:-1] * (kwargs['level']*4/20+1)
            freqs[c][np.abs(freqs[c]) < thres] = 0
            freqs[c][fs_list[20]:] = np.around(freqs[c][fs_list[20]:] / rounder[fs_list[20]:]) * rounder[fs_list[20]:]

        return freqs

    get_range = lambda fs, sr, x: x is not np.inf and int(fs*x*2/sr+0.5) or 2**32
