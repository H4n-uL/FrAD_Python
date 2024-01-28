import numpy as np
import struct, base64, os

class dsd:
    def build_dff_header(datalen, channels, sample_rate):
        CMPR = base64.b64decode('RFNEIA9ub3QgY29tcHJlc3NlZAA=')

        PROP = \
            b'SND ' + \
            b'FS  ' + struct.pack('>Q', 4) + struct.pack('>I', sample_rate) + \
            b'CHNL' + struct.pack('>Q', 2+len(b''.join(channels))) + struct.pack('>H', len(channels)) + b''.join(channels) + \
            b'CMPR' + struct.pack('>Q', len(CMPR)) + CMPR

        HEAD = bytearray(\
            b'FRM8' + \
            b'\x00'*8 + \
            b'DSD ' + \
            b'FVER' + struct.pack('>Q', 4) + b'\x01\x05\x00\x00' + \
            b'PROP' + struct.pack('>Q', len(PROP)) + PROP + \
            b'DSD ' + struct.pack('>Q', datalen))
        
        HEAD[0x4:0xc] = struct.pack('>Q', len(HEAD) + datalen)
        return bytes(HEAD)

    def delta_sigma_msbf(x):
        integrator_state, quantizer_state = 0, 0
        bitstream = np.zeros_like(x)
        
        for i in range(len(x)):
            integrator_state += x[i] - quantizer_state
            quantizer_state = 1 if integrator_state > 0 else -1
            bitstream[i] = 1 if quantizer_state==1 else 0
        
        return np.packbits([int(b)for b in bitstream])

if __name__ == '__main__':
    pcm_name = 'the.pcm'
    dff_name = 'output.dff'

    channels_batch = [b'SLFT', b'SRGT']

    with open(pcm_name, 'rb') as pcm, open('temp.dsd', 'wb') as temp:
        while True:
            block = pcm.read(4 * len(channels_batch) * 1048576)
            data_numpy = np.frombuffer(block, dtype=np.int32).astype(np.float64) / 2**32
            freq = [data_numpy[i::len(channels_batch)] for i in range(len(channels_batch))]
            block = np.column_stack([dsd.delta_sigma_msbf(c).astype(np.uint8) for c in freq]).ravel(order='C').tobytes()
            temp.write(block)
            dlen = os.path.getsize('temp.dsd')
            with open('temp.dsd', 'rb') as trd, open(dff_name, 'wb') as dsdfile:
                dsdfile.write(dsd.build_dff_header(dlen, channels_batch, 2822400) + trd.read())
