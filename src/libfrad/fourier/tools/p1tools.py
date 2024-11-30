import numpy as np
import struct

# Modified Opus Subbands
MOS =  (0,     200,   400,   600,   800,   1000,  1200,  1400,
        1600,  2000,  2400,  2800,  3200,  4000,  4800,  5600,
        6800,  8000,  9600,  12000, 15600, 20000, 24000, 28800,
        34400, 40800, 48000, (2**32)-1)

subbands = len(MOS) - 1
rndint = lambda x: int(x+0.5)
spread_alpha = 0.8
quant_alpha = 0.75

def get_bin_range(dlen: int, srate: int, subband_index: int) -> slice:
    return slice(rndint(dlen/(srate/2)*MOS[subband_index]), rndint(dlen/(srate/2)*MOS[subband_index+1]))

def mask_thres_mos(freqs: np.ndarray, srate: int, bit_depth: int, loss_level: float, alpha: float) -> np.ndarray:
    freqs = np.abs(freqs)
    pcm_scale = 1 << (bit_depth-1)
    thres = np.zeros(subbands)
    for i in range(subbands):
        subfreqs = freqs[get_bin_range(len(freqs), srate, i)]
        if len(subfreqs) == 0: break

        f = (MOS[i] + MOS[i+1]) / 2
        absolute_hearing_threshold = 10.0**(
            (3.64*(f/1000.)**-0.8 - 6.5*np.exp(-0.6*(f/1000.-3.3)**2.) + 1e-3*((f/1000.)**4.))/20
        ) / pcm_scale

        sfq = np.sqrt(np.mean(subfreqs**2)) ** alpha
        thres[i] = np.maximum(sfq, min(absolute_hearing_threshold, 1.0)) * loss_level

    return thres

def mapping_from_opus(mapped_thres, freqs_len, srate):
    thres = np.zeros(freqs_len)
    for i in range(subbands-1):
        start = min(get_bin_range(freqs_len, srate, i).start, freqs_len)
        end = min(get_bin_range(freqs_len, srate, i+1).start, freqs_len)
        thres[start:end] = np.linspace(mapped_thres[i], mapped_thres[i+1], end-start)
    return thres

def quant(x: np.ndarray) -> np.ndarray: return np.sign(x) * np.abs(x)**quant_alpha
def dequant(x: np.ndarray) -> np.ndarray: return np.sign(x) * np.abs(x)**(1/quant_alpha)

bitstr2bytes = lambda bstr: bytes(int(bstr[i:i+8].ljust(8, '0'), 2) for i in range(0, len(bstr), 8))
bytes2bitstr = lambda b: ''.join(f'{byte:08b}' for byte in b)

def exp_golomb_rice_encode(data: np.ndarray):
    if not data.size: return b'\x00'
    dmax = np.abs(data).max()
    k = dmax and int(np.ceil(np.log2(dmax))) or 0
    encoded = ''
    for n in data:
        n = (n>0) and (2*n-1) or (-2*n)
        binary_code = bin(n + 2**k)[2:]
        m = len(binary_code) - (k+1)
        encoded += ('0' * m + binary_code)

    return struct.pack('B', k) + bitstr2bytes(encoded)

def exp_golomb_rice_decode(dbytes: bytes):
    k = struct.unpack('B', dbytes[:1])[0]
    decoded = []
    data = bytes2bitstr(dbytes[1:])
    while data:
        try: m = data.index('1')
        except: break

        codeword, data = data[:(m*2)+k+1], data[(m*2)+k+1:]
        n = int(codeword, 2) - 2**k
        decoded.append((n+1)//2 if n%2==1 else -n//2)

    return np.array(decoded)
