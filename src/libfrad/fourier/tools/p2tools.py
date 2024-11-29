import numpy as np
from scipy import signal

MAX_ORDER = 12
COEF_RES = 4
MIN_PRED = np.log10(2) * 10

def calc_autocorr(signal: np.ndarray) -> np.ndarray:
    window = np.exp(-0.5 * (np.arange(MAX_ORDER + 1) * 0.4)**2)
    return window * np.correlate(signal, signal, mode='full')[len(signal)-1:len(signal)+MAX_ORDER]

def levinson_durbin(autocorr: np.ndarray) -> np.ndarray:
    lpc = np.zeros(MAX_ORDER + 1)
    lpc[0] = 1.0
    error = autocorr[0]

    if error <= 0: return np.zeros(MAX_ORDER + 1)

    for i in range(1, MAX_ORDER + 1):
        reflection = -np.sum(lpc[:i] * autocorr[i:0:-1])
        if error < 1e-9: break

        reflection /= error
        if abs(reflection) >= 1.0: break

        lpc[i] = reflection
        for j in range(1, i): lpc[j] += reflection * lpc[i-j]
        error *= 1.0 - reflection * reflection
        if error <= 0: break

    return lpc

def quantise_lpc(lpc: np.ndarray) -> np.ndarray:
    eps = 1e-6
    return np.clip(
        lpc * ((1<<(COEF_RES-1)) - 1),
        -(1<<(COEF_RES-1)) + eps,
        (1<<(COEF_RES-1)) - 1 - eps
    ).round().astype(int)

def dequantise_lpc(lpcq):
    return lpcq.astype(float) / ((1<<(COEF_RES-1)) - 1)

def predgain(orig, prc):
    orig_energy = np.sum(np.abs(orig)**2)
    if orig_energy < 1e-9: return 0
    error_energy = np.sum(np.abs(orig - prc)**2)
    if error_energy < 1e-9: return 0
    return 10 * np.log10(orig_energy / error_energy)

def tns_analysis(freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tns_freqs = np.zeros_like(freqs)
    lpcqs = []

    for i in range(len(freqs)):
        autocorr = calc_autocorr(freqs[i])
        lpc = levinson_durbin(autocorr)

        if np.any(np.abs(lpc) >= 1.0):
            tns_freqs[i] = freqs[i]
            lpcqs.append(np.zeros(MAX_ORDER+1))
            continue

        lpcq = quantise_lpc(lpc)
        lpcdeq = dequantise_lpc(lpcq)

        # Use filtered signal only if filter is stable
        filtered = signal.lfilter(lpcdeq, [1], freqs[i])
        if np.isnan(filtered).any() or np.isinf(filtered).any() or predgain(freqs[i], filtered) < MIN_PRED:
            tns_freqs[i] = freqs[i]
            lpcqs.append(np.zeros(MAX_ORDER+1))
        else:
            tns_freqs[i] = filtered
            lpcqs.append(lpcq)

    return tns_freqs, np.array(lpcqs)

def tns_synthesis(tns_freqs: np.ndarray, lpcqs: np.ndarray) -> np.ndarray:
    freqs = np.zeros_like(tns_freqs)

    for i in range(len(tns_freqs)):
        if np.all(lpcqs[i] == 0):
            freqs[i] = tns_freqs[i]
            continue

        lpcdeq = dequantise_lpc(lpcqs[i])

        filtered = signal.lfilter([1], lpcdeq, tns_freqs[i])
        if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
            freqs[i] = tns_freqs[i]
        else: freqs[i] = filtered

    return freqs