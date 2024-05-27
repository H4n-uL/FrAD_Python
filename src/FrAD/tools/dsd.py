import base64, os, struct, subprocess, time
from ..common import variables, methods
import numpy as np

class DeltaSigma:
    def __init__(self, deg=2):
        self.deg = deg
        self.intg = [0]*deg
        self.quant = 0

    def modulator(self, x):
        assert self.deg > 0, 'ΔΣ modulator degree should be greater than 0.'
        bitstream = np.zeros_like(x)

        for i in range(len(x)):
            self.intg[0] += x[i] - self.quant
            self.intg[1:self.deg] += self.intg[:self.deg-1] - self.quant
            self.quant = 1 if self.intg[-1] > 0 else -1
            bitstream[i] = 1 if self.quant==1 else 0

        return np.packbits([int(b) for b in bitstream.astype(np.uint8)])

    def demodulator(self, bitstream):
        bitstream = np.unpackbits(bitstream)
        y = np.zeros_like(bitstream, dtype=np.float64)

        for i in range(len(bitstream)):
            self.quant = 1 if bitstream[i] == 1 else -1
            for j in range(self.deg-1, 0, -1):
                self.intg[j] = self.intg[j-1]
            self.intg[0] = self.quant
            y[i] = self.intg[-1]

        return y

class dsd:
    channels_dict = {
        1: [b'SLFT'],
        2: [b'SLFT', b'SRGT'],
        3: [b'MLFT', b'MRGT', b'C   '],
        4: [b'MLFT', b'MRGT', b'LS  ', b'RS  '],
        5: [b'MLFT', b'MRGT', b'C   ', b'LS  ', b'RS  '],
        6: [b'MLFT', b'MRGT', b'C   ', b'LFE ', b'LS  ', b'RS  ']
    }

    @staticmethod
    def build_dff_header(datalen: int, channels: list, smprate: int):
        CMPR = base64.b64decode('RFNEIA9ub3QgY29tcHJlc3NlZAA=')

        PROP = bytes(
            b'SND ' + \
            b'FS  ' + struct.pack('>Q', 4) + struct.pack('>I', smprate) +
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

    @staticmethod
    def build_dsf_header(datalen: int, chtype: int, smprate: int, dsfblock: int):
        channels = chtype - 1 if chtype > 4 else chtype
        FMT = bytearray(
            b'fmt ' + struct.pack('<Q', 52) +
            struct.pack('<I', 1) +                        # Version
            struct.pack('<I', 0) +                        # Format ID
            struct.pack('<I', chtype) +                   # Channel Type
            struct.pack('<I', channels) +
            struct.pack('<I', smprate) +
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

    @staticmethod
    def encode(f, srate, channels, out, ext, verbose: bool = False):
        chb = dsd.channels_dict[channels]

        cli_width = 40
        i = 0

        dsd_srate = 2822400
        pred_size = os.path.getsize(f) // srate * dsd_srate // 64
        try:
            BUFFER_SIZE = 262144 * 8 * channels

            delta_sigma = [DeltaSigma() for _ in range(channels)]
            command = [
                variables.ffmpeg,
                '-v', 'quiet',
                '-f', 'f64be',
                '-ar', str(srate),
                '-ac', str(channels),
                '-i', f,
                '-ar', str(dsd_srate),
                '-f', 'f64be',
                'pipe:1']

            pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            if pipe.stdout is None: raise Exception('Broken pipe.')

            h = b''
            with open(variables.temp_dsd, 'wb') as bstr:
                if verbose: print('\n\n')
                start_time = time.time()
                while True:
                    i += BUFFER_SIZE
                    data = pipe.stdout.read(BUFFER_SIZE)
                    if not data or data == b'': break
                    # print(data)
                    data_numpy = np.frombuffer(data, dtype='>f8') / 2
                    freq = [data_numpy[i::channels] for i in range(channels)]
                    block = np.column_stack([delta_sigma[c].modulator(freq[c]) for c in range(len(freq))]).ravel(order='C').tobytes()

                    bstr.write(block)
                    dlen = os.path.getsize(variables.temp_dsd)
                    h = dsd.build_dff_header(dlen, chb, dsd_srate)

                    if verbose:
                        elapsed_time = time.time() - start_time
                        bps = i / elapsed_time
                        mult = (dlen * 8 / dsd_srate / channels) / elapsed_time
                        percent = (dlen / pred_size)*100
                        b = int(percent / 100 * cli_width)
                        eta = (elapsed_time / (percent / 100)) - elapsed_time if percent != 0 else 'infinity'
                        print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                        print(f'DSD Encode Speed: {(bps / 10**6):.3f} MB/s, X{mult:.3f}')
                        print(f'elapsed: {methods.tformat(elapsed_time)}, ETA {methods.tformat(eta)}')
                        print(f"[{'█'*b}{' '*(cli_width-b)}] {percent:.3f}% completed")
                with open(f'{out}.{ext}', 'wb') as f, open(variables.temp_dsd, 'rb') as temp:
                    f.write(h + temp.read())
                if verbose: print('\x1b[1A\x1b[2K', end='')
        except KeyboardInterrupt: bstr.close()
        finally:
            pipe.kill()
            dlen = os.path.getsize(variables.temp_dsd)
            h = dsd.build_dff_header(dlen, chb, dsd_srate)
            with open(f'{out}.{ext}', 'wb') as f, open(variables.temp_dsd, 'rb') as temp:
                f.write(h + temp.read())
