import numpy as np

def sbr_encode(freqs: np.ndarray):
    lf = len(freqs)//8
    A = [freqs[i:i+lf] for i in range(0, len(freqs), lf)]
    B = [np.sqrt(np.mean(np.abs(i)**2)) for i in A]
    B_max = np.max(B)
    B = np.around(B / B_max * 255)

    return A[0], B
