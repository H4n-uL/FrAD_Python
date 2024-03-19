import numpy as np

class psycho:
    def f_SP_dB(maxfreq,nfilts):
        maxbark = psycho.hz2bark(maxfreq)
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
        maxbark=psycho.hz2bark(maxfreq)
        step_bark = maxbark/(nfilts-1)
        barks=np.arange(0,nfilts)*step_bark
        f=psycho.bark2hz(barks)+1e-6
        LTQ=np.clip((3.64*(f/1000.)**-0.8 -6.5*np.exp(-0.6*(f/1000.-3.3)**2.)
            +1e-3*((f/1000.)**4.)),-20,120)
        mTbark=np.max((mTbark, 10.0**((LTQ-60)/20)),0)
        return mTbark

    def hz2bark(f): return 6. * np.arcsinh(f/600.)

    def bark2hz(Brk): return 600. * np.sinh(Brk/6.)

    def mapping2barkmat(fs, nfilts, nfft):
        maxbark = psycho.hz2bark(fs/2)
        step_bark = maxbark/(nfilts-1)
        binbark = psycho.hz2bark(np.linspace(0,(nfft/2),int(nfft/2)+1)*fs/nfft)
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
