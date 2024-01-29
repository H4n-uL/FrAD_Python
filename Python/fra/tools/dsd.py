import base64, os, struct, time
from ..common import variables, methods
import numpy as np

class dsd:
    def delta_sigma(x, deg=2):
        assert deg > 0, 'ΔΣ modulator degree should be greater than 0.'

        intg = [0]*deg
        quant = 0
        bitstream = np.zeros_like(x)

        for i in range(len(x)):
            intg[0] += x[i] - quant
            for j in range(1, deg):
                intg[j] += intg[j-1] - quant
            quant = 1 if intg[-1] > 0 else -1
            bitstream[i] = 1 if quant==1 else 0

        return np.packbits([int(b) for b in bitstream.astype(np.uint8)])

    channels_dict = {
        1: [b'SLFT'],
        2: [b'SLFT', b'SRGT'],
        3: [b'MLFT', b'MRGT', b'C   '],
        4: [b'MLFT', b'MRGT', b'LS  ', b'RS  '],
        5: [b'MLFT', b'MRGT', b'C   ', b'LS  ', b'RS  '],
        6: [b'MLFT', b'MRGT', b'C   ', b'LFE ', b'LS  ', b'RS  ']
    }

    def build_dff_header(datalen: int, channels: int, sample_rate: int):
        CMPR = base64.b64decode('RFNEIA9ub3QgY29tcHJlc3NlZAA=')

        PROP = bytes(
            b'SND ' + \
            b'FS  ' + struct.pack('>Q', 4) + struct.pack('>I', sample_rate) +
            b'CHNL' + struct.pack('>Q', 2+len(b''.join(channels))) + struct.pack('>H', len(channels)) + b''.join(channels) +
            b'CMPR' + struct.pack('>Q', len(CMPR)) + CMPR
        )

        HEAD = bytearray(\
            b'FRM8' + \
            b'\x00'*8 + \
            b'DSD ' + \
            b'FVER' + struct.pack('>Q', 4) + b'\x01\x05\x00\x00' + \
            b'PROP' + struct.pack('>Q', len(PROP)) + PROP + \
            b'DSD ' + struct.pack('>Q', datalen)
        )

        HEAD[0x4:0xc] = struct.pack('>Q', len(HEAD) + datalen)
        return bytes(HEAD)

    def build_dsf_header(datalen: int, chtype: int, sample_rate: int, dsfblock: int):
        channels = chtype - 1 if chtype > 4 else chtype
        FMT = bytearray(
            b'fmt ' + struct.pack('<Q', 52) + 
            struct.pack('<I', 1) +                        # Version
            struct.pack('<I', 0) +                        # Format ID
            struct.pack('<I', chtype) +                   # Channel Type
            struct.pack('<I', channels) +
            struct.pack('<I', sample_rate) +
            struct.pack('<I', 8) +                        # Sample bits
            struct.pack('<Q', datalen * 8 // channels) +  # Sample count
            struct.pack('<I', dsfblock) +                 # Block size / channel
            struct.pack('<I', 0)                          # Reserved
        )
        FMT[0x4:0xc] = struct.pack('<Q', len(FMT))
        FMT = bytes(FMT)

        DATA = bytes(
            b'data' +
            struct.pack('<Q', datalen + 12)
        )

        HEAD = bytearray(
            b'DSD ' + 
            struct.pack('<Q', 28) + 
            struct.pack('<Q', 0) + # Data length padding
            struct.pack('<Q', 0) + # Metadata location
            FMT + DATA
        )
        HEAD[0xc:0x14] = struct.pack('<Q', len(HEAD) + datalen)
        return bytes(HEAD)

    def encode(srate, channels, bits, out, ext, verbose: bool = False):
        chb = dsd.channels_dict[channels]

        plen = os.path.getsize(variables.temp_pcm)
        cli_width = 40
        i = 0

        dsd_srate = 2822400
        try:
            with open(variables.temp_pcm, 'rb') as pcm, open(variables.temp_dsd, 'wb') as temp:
                start_time = time.time()
                if verbose: print('\n')

                while True:
                    block = pcm.read(bits // 8 * len(chb) * srate)
                    if not block: break
                    i += len(block)
                    if bits == 32:   data_numpy = np.frombuffer(block, dtype=np.int32)
                    elif bits == 16: data_numpy = np.frombuffer(block, dtype=np.int16)
                    elif bits == 8:  data_numpy = np.frombuffer(block, dtype=np.uint8)
                    data_numpy = data_numpy.astype(np.float64) / 2**(bits-1)

                    freq = [data_numpy[i::len(chb)] for i in range(len(chb))]
                    block = np.column_stack([dsd.delta_sigma(methods.resample_1sec(c, srate, dsd_srate)) for c in freq]).ravel(order='C').tobytes()
                    temp.write(block)

                    dlen = os.path.getsize(variables.temp_dsd)
                    with open(variables.temp_dsd, 'rb') as trd, open(f'{out}.{ext}', 'wb') as dsdfile:
                        dsdfile.write(dsd.build_dff_header(dlen, chb, dsd_srate) + trd.read())

                    if verbose:
                        elapsed_time = time.time() - start_time
                        bps = i / elapsed_time
                        mult = dlen / bits * 64 / dsd_srate / channels / elapsed_time
                        percent = i*100 / plen
                        b = int(percent / 100 * cli_width)
                        print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                        print(f'DSD Encode Speed: {(bps / 10**6):.3f} MB/s, X{mult:.3f}')
                        print(f"[{'█'*b}{' '*(cli_width-b)}] {percent:.3f}% completed")
                if verbose: print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
        except KeyboardInterrupt: pass
        finally:
            os.remove(variables.temp_pcm)
            os.remove(variables.temp_dsd)
