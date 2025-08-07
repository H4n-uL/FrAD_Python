import numpy as np
import struct

MODIFIED_OPUS_SUBBANDS = (
    0,     200,   400,   600,   800,   1000,  1200,  1400,
    1600,  2000,  2400,  2800,  3200,  4000,  4800,  5600,
    6800,  8000,  9600,  12000, 15600, 20000, 24000, 28800,
    34400, 40800, 48000, (2**32)-1
)

SUBBANDS = len(MODIFIED_OPUS_SUBBANDS) - 1
SPREAD_ALPHA = 0.8
QUANT_ALPHA = 0.75

def get_bin_range(dlen: int, srate: int, subband_index: int) -> slice:
    return slice(round(dlen/(srate/2)*MODIFIED_OPUS_SUBBANDS[subband_index]), round(dlen/(srate/2)*MODIFIED_OPUS_SUBBANDS[subband_index+1]))

def mask_thres_mos(freqs: np.ndarray, srate: int, pcm_factor: float, loss_level: float, alpha: float) -> np.ndarray:
    freqs = np.abs(freqs)
    thres = np.zeros(SUBBANDS)
    for i in range(SUBBANDS):
        subfreqs = freqs[get_bin_range(len(freqs), srate, i)]
        if len(subfreqs) == 0: break

        f = (MODIFIED_OPUS_SUBBANDS[i] + MODIFIED_OPUS_SUBBANDS[i+1]) / 2
        absolute_hearing_threshold = 10.0**(
            (3.64 * (f / 1000.0) ** -0.8 - 6.5 * np.exp(-0.6 * (f / 1000.0 - 3.3) ** 2.0) + 1e-3 * ((f / 1000.0) ** 4.0)) / 20
        )

        sfq = np.sqrt(np.mean(subfreqs**2)) ** alpha * np.sqrt(pcm_factor)
        thres[i] = np.maximum(sfq, absolute_hearing_threshold) * loss_level

    return thres

def mapping_from_opus(mapped_thres, freqs_len, srate):
    thres = np.zeros(freqs_len)
    for i in range(SUBBANDS-1):
        start = min(get_bin_range(freqs_len, srate, i).start, freqs_len)
        end = min(get_bin_range(freqs_len, srate, i+1).start, freqs_len)
        thres[start:end] = np.linspace(mapped_thres[i], mapped_thres[i+1], end-start)
    return thres

def quant(x: np.ndarray) -> np.ndarray: return np.sign(x) * np.abs(x)**QUANT_ALPHA
def dequant(x: np.ndarray) -> np.ndarray: return np.sign(x) * np.abs(x)**(1/QUANT_ALPHA)

bitstr2bytes = lambda bstr: bytes(int(bstr[i:i+8].ljust(8, '0'), 2) for i in range(0, len(bstr), 8))
bytes2bitstr = lambda b: ''.join(f'{byte:08b}' for byte in b)

def exp_golomb_rice_encode(data: np.ndarray):
    if not data.size: return b'\x00'
    dmax = np.abs(data).max()
    k = dmax and int(np.ceil(np.log2(dmax))) or 0
    encoded = ''
    for n in data:
        n = ((n << 1) - 1) if (n > 0) else (-n << 1)
        binary_code = bin(n + (1 << k))[2:]
        m = len(binary_code) - (k + 1)
        encoded += ('0' * m + binary_code)

    return struct.pack('B', k) + bitstr2bytes(encoded)

def exp_golomb_rice_decode(dbytes: bytes):
    k = struct.unpack('B', dbytes[:1])[0]
    decoded = []
    data = bytes2bitstr(dbytes[1:])
    while data:
        try: m = data.index('1')
        except: break

        codeword, data = data[:(m * 2) + k + 1], data[(m * 2) + k + 1:]
        n = int(codeword, 2) - (1 << k)
        decoded.append((n + 1) >> 1 if n & 1 == 1 else -(n >> 1))

    return np.array(decoded)
