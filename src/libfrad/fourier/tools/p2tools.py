import numpy as np
from scipy import signal

MAX_ORDER = 12
COEF_RES = 4
MIN_PRED = np.log10(2) / 10

def calc_autocorr(signal: np.ndarray) -> np.ndarray:
    sig = signal - np.mean(signal)
    norm = np.sqrt(np.sum(sig**2))
    if norm > 1e-6: sig = sig / norm
    full_corr = np.correlate(sig, sig, mode='full')
    autocorr = full_corr[len(sig)-1:len(sig)+MAX_ORDER]
    window = np.exp(-0.5 * (np.arange(MAX_ORDER + 1) * 0.01)**2)
    return autocorr * window

def levinson_durbin(autocorr: np.ndarray) -> np.ndarray:
    lpc = np.zeros(MAX_ORDER + 1)
    lpc[0] = 1.0
    error = autocorr[0]
    if error <= 1e-10: return lpc

    for i in range(1, MAX_ORDER + 1):
        reflection = -np.sum(lpc[:i] * autocorr[i:0:-1]) / error
        if abs(reflection) >= 0.96: reflection = 0.96 * np.sign(reflection)

        lpc_tmp = lpc.copy()
        lpc[i] = reflection
        for j in range(1, i): 
            lpc[j] += reflection * lpc_tmp[i-j]
        error *= 1.0 - reflection * reflection
        if error <= 1e-12: break

    return lpc

def quantise_lpc(lpc: np.ndarray) -> np.ndarray:
    scale = (1 << COEF_RES) - 1

    lpc_quant = np.zeros_like(lpc, dtype=int)
    if len(lpc) > 1:
        lpc_quant[1:] = np.clip(
            lpc[1:] * scale,
            -scale,
            scale - 1
        ).round().astype(int)

    return lpc_quant

def dequantise_lpc(lpc_quant: np.ndarray) -> np.ndarray:
    if np.all(lpc_quant == 0): return np.array([1.0])
    scale = (1 << COEF_RES) - 1
    lpc_deq = np.zeros_like(lpc_quant, dtype=float)
    lpc_deq[0] = 1.0
    if len(lpc_quant) > 1: lpc_deq[1:] = lpc_quant[1:].astype(float) / scale
    return lpc_deq

def predgain(orig: np.ndarray, residual: np.ndarray) -> float:
    orig_centred = orig - np.mean(orig)
    resid_centred = residual - np.mean(residual)

    orig_energy = np.sum(orig_centred**2)
    resid_energy = np.sum(resid_centred**2)

    if (
        orig_energy < 1e-10 or
        resid_energy < 1e-10 or
        resid_energy >= orig_energy
    ): return 0

    return 20 * np.log10(orig_energy / resid_energy)

def tns_analysis(freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lpc_zero = np.zeros(MAX_ORDER + 1)
    if len(freqs) < MAX_ORDER * 2 or not lpc_cond(freqs):
        return freqs, lpc_zero

    energy = np.sum(freqs**2)
    if energy < 1e-10: return freqs, lpc_zero

    autocorr = calc_autocorr(freqs)
    lpc = levinson_durbin(autocorr)

    if np.sum(np.abs(lpc[1:])) < 0.01: return freqs, lpc_zero
    lpc_quant = quantise_lpc(lpc)
    if np.all(lpc_quant[1:] == 0): return freqs, lpc_zero
    lpc_deq = dequantise_lpc(lpc_quant)

    residual = signal.lfilter(lpc_deq, [1], freqs)
    max_val = np.max(np.abs(residual))
    if np.isnan(residual).any() or np.isinf(residual).any() or max_val > 1e6:
        return freqs, lpc_zero

    gain = predgain(freqs, residual)
    if gain < MIN_PRED: return freqs, lpc_zero
    return residual, lpc_quant

def tns_synthesis(tns_freqs: np.ndarray, lpc_quant: np.ndarray) -> np.ndarray:
    if np.all(lpc_quant == 0): return tns_freqs
    lpc_deq = dequantise_lpc(lpc_quant)
    filtered = signal.lfilter([1], lpc_deq, tns_freqs)

    max_val = np.max(np.abs(filtered))
    if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)) or max_val > 1e6:
        return tns_freqs

    return filtered

def lpc_cond(freqs):
    geo_mean = np.exp(np.mean(np.log(np.abs(freqs) + 1e-10)))
    arith_mean = np.mean(np.abs(freqs))
    return geo_mean / (arith_mean + 1e-10) < 0.5
