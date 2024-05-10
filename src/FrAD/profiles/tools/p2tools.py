import numpy as np

def sbr_encode(freqs: np.ndarray, channels: int):
    freqs = np.log10(np.abs(freqs))*20
    BR = []
    for c in range(channels):
        lf = len(freqs[c])//8
        A = [freqs[c][i:i+lf] for i in range(0, len(freqs[c]), lf)]
        B = np.array([np.mean(i) for i in A])
        BR.append(np.around(B - np.max(B) + 255))
    print(BR)

    return np.array(BR)

def sbr_decode(freqs: np.ndarray, channels: int, gains: np.ndarray):
    return np.array([np.array([freqs[c] * 10**((gains[c][i]+gains[c][0]-255)/20) for i in range(8)]).ravel() for c in range(channels)])
