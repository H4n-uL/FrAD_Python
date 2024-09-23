import numpy as np
from scipy import signal

bitstr2bytes = lambda bstr: bytes(int(bstr[i:i+8].ljust(8, '0'), 2) for i in range(0, len(bstr), 8))
bytes2bitstr = lambda b: ''.join(f'{byte:08b}' for byte in b)

class tns:
    MAX_ORDER = 12
    COEF_RES = 4
    MIN_PRED = np.log10(2) * 20

    @staticmethod
    def calc_autocorr(signal):
        return np.correlate(signal, signal, mode='full')[len(signal)-1:len(signal)+tns.MAX_ORDER]

    @staticmethod
    def levinson_durbin(autocorr):
        lpc = np.zeros(tns.MAX_ORDER + 1)
        lpc[0] = 1.0
        error = autocorr[0]
        for i in range(1, tns.MAX_ORDER + 1):
            reflection = -np.sum(lpc[:i] * autocorr[i:0:-1]) / error
            lpc[:i] = lpc[:i] + reflection * lpc[i-1::-1]
            lpc[i] = reflection
            error *= 1 - reflection**2
        return lpc

    @staticmethod
    def quantize_lpc(lpc):
        scaled_lpc = lpc * (2**(tns.COEF_RES-1) - 1)
        lpcq = np.clip(np.round(scaled_lpc), -(2**(tns.COEF_RES-1)), 2**(tns.COEF_RES-1)-1).astype(int)
        return lpcq

    @staticmethod
    def dequantize_lpc(lpcq):
        return lpcq.astype(float) / (2**(tns.COEF_RES-1) - 1)

    @staticmethod
    def predgain(orig, prc):
        return 10 * np.log10(np.sum(np.abs(orig)**2) / np.sum(np.abs(orig - prc)**2))

    @staticmethod
    def analysis(freqs: np.ndarray):
        tns_freqs = np.zeros_like(freqs)
        lpcqs = []
        for i in range(len(freqs)):
            autocorr = tns.calc_autocorr(freqs[i])
            lpc = tns.levinson_durbin(autocorr)
            lpcq = tns.quantize_lpc(lpc)
            lpcdeq = tns.dequantize_lpc(lpcq)
            tns_processed = signal.lfilter(lpcdeq, [1], freqs[i])
            if tns.predgain(freqs, tns_freqs) < tns.MIN_PRED:
                tns_freqs[i] = freqs[i]
                lpcqs.append(np.array([0]*(tns.MAX_ORDER+1)))
            else:
                tns_freqs[i] = tns_processed
                lpcqs.append(lpcq)

        return tns_freqs, np.array(lpcqs)

    @staticmethod
    def synthesis(tns_freqs, lpcqs):
        freqs = np.zeros_like(tns_freqs)
        for i in range(len(tns_freqs)):
            if np.all(lpcqs[i] == 0): return tns_freqs
            lpcdeq = tns.dequantize_lpc(lpcqs[i])
            freqs[i] = signal.lfilter([1], lpcdeq, tns_freqs[i])

        return freqs

class pns:
    MIN_PRED = np.log10(2) * 20
    NOISE_FLOOR = -60
    HIGH_FREQ_CUTOFF = 0.9

    @staticmethod
    def calc_band_energy(freqs):
        return np.mean(np.abs(freqs)**2)

    @staticmethod
    def noise_detection(freqs):
        band_energy = pns.calc_band_energy(freqs)
        noise_measure = 1.0 / (1.0 + np.var(freqs) / (band_energy + 1e-10))
        return noise_measure

    @staticmethod
    def generate_noise(size, energy):
        noise = np.random.normal(0, np.sqrt(energy), size)
        noise_energy = pns.calc_band_energy(noise)
        return noise * np.sqrt(energy / noise_energy)

    @staticmethod
    def analysis(freqs: np.ndarray):
        pns_freqs = np.zeros_like(freqs, dtype=int)
        high_freq_energies = []

        for i in range(len(freqs)):
            n = len(freqs[i])
            cutoff_index = int(n * pns.HIGH_FREQ_CUTOFF)
            noise_measure = pns.noise_detection(freqs[i][cutoff_index:])
            high_freq_energy = 0

            if noise_measure > 0.5:
                high_freq_energy = pns.calc_band_energy(freqs[i][cutoff_index:])
                noise_floor = 10**(pns.NOISE_FLOOR/10) * high_freq_energy

                pns_freqs[i] = freqs[i]
                pns_freqs[i][cutoff_index:] = pns.generate_noise(n - cutoff_index, noise_floor)

                if pns.predgain(freqs[i], pns_freqs[i]) < pns.MIN_PRED: high_freq_energy = 0

            pns_freqs[i] = np.around(pns_freqs[i])
            high_freq_energies.append(high_freq_energy)

        return pns_freqs, np.array(high_freq_energies)

    @staticmethod
    def synthesis(pns_freqs: np.ndarray, high_freq_energy: np.ndarray):
        rec_freqs = np.zeros_like(pns_freqs)

        for i in range(len(pns_freqs)):
            if high_freq_energy[i] == 0:
                rec_freqs[i] = pns_freqs[i]
                continue
            n = len(pns_freqs[i])
            cutoff_index = int(n * pns.HIGH_FREQ_CUTOFF)
            pns_freqs[i][cutoff_index:] = pns.generate_noise(n - cutoff_index, high_freq_energy[i])
            rec_freqs[i] = pns_freqs[i]

        return rec_freqs

    @staticmethod
    def predgain(orig, prc):
        return 10 * np.log10(np.sum(np.abs(orig)**2) / np.sum(np.abs(orig - prc)**2))
