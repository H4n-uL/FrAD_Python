import numpy as np
from scipy import signal

bitstr2bytes = lambda bstr: bytes(int(bstr[i:i+8].ljust(8, '0'), 2) for i in range(0, len(bstr), 8))
bytes2bitstr = lambda b: ''.join(f'{byte:08b}' for byte in b)

MAX_ORDER = 12
COEF_RES = 4
MIN_PRED = np.log10(2) * 20

def calc_autocorr(signal):
    return np.correlate(signal, signal, mode='full')[len(signal)-1:len(signal)+MAX_ORDER]

def levinson_durbin(autocorr):
    lpc = np.zeros(MAX_ORDER + 1)
    lpc[0] = 1.0
    error = autocorr[0]
    for i in range(1, MAX_ORDER + 1):
        reflection = -np.sum(lpc[:i] * autocorr[i:0:-1]) / error
        lpc[:i] = lpc[:i] + reflection * lpc[i-1::-1]
        lpc[i] = reflection
        error *= 1 - reflection**2
    return lpc

def quantise_lpc(lpc):
    scaled_lpc = lpc * ((1<<(COEF_RES-1)) - 1)
    lpcq = np.clip(np.round(scaled_lpc), -(1<<(COEF_RES-1)), (1<<(COEF_RES-1))-1).astype(int)
    return lpcq

def dequantise_lpc(lpcq):
    return lpcq.astype(float) / ((1<<(COEF_RES-1)) - 1)

def predgain(orig, prc):
    return 10 * np.log10(np.sum(np.abs(orig)**2) / np.sum(np.abs(orig - prc)**2))

def tns_analysis(freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tns_freqs = np.zeros_like(freqs)
    lpcqs = []
    for i in range(len(freqs)):
        autocorr = calc_autocorr(freqs[i])
        lpc = levinson_durbin(autocorr)
        lpcq = quantise_lpc(lpc)
        lpcdeq = dequantise_lpc(lpcq)
        tns_freqs[i] = signal.lfilter(lpcdeq, [1], freqs[i])
        if predgain(freqs, tns_freqs[i]) < MIN_PRED:
            tns_freqs[i] = freqs[i]
            lpcqs.append(np.array([0]*(MAX_ORDER+1)))
        else: lpcqs.append(lpcq)

    return tns_freqs, np.array(lpcqs)

def tns_synthesis(tns_freqs, lpcqs):
    freqs = np.zeros_like(tns_freqs)
    for i in range(len(tns_freqs)):
        if np.all(lpcqs[i] == 0): freqs[i] = tns_freqs[i]; continue
        lpcdeq = dequantise_lpc(lpcqs[i])
        freqs[i] = signal.lfilter([1], lpcdeq, tns_freqs[i])

    return freqs
