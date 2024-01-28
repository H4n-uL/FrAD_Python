import base64, os, secrets, struct
import numpy as np

class dsd:
    def delta_sigma(x):
        integrator_state, quantizer_state = 0, 0
        bitstream = np.zeros_like(x)

        for i in range(len(x)):
            integrator_state += x[i] - quantizer_state
            quantizer_state = 1 if integrator_state > 0 else -1
            bitstream[i] = 1 if quantizer_state==1 else 0
        
        return np.packbits([int(b)for b in bitstream])

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

    def build_dsf_header(datalen: int, chtype: int, sample_rate: int):
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
            struct.pack('<I', 1) +                        # Block size / channel
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

if __name__ == '__main__':
    temp_file = f'temp.{base64.b64encode(secrets.token_bytes(6)).decode().replace("/", "_")}.dsd'
    pcm_name = 'input.pcm'
    dsd_name = 'output.dff'

    channels_batch = [b'SLFT', b'SRGT']

    # DFF #
    try:
        with open(pcm_name, 'rb') as pcm, open(temp_file, 'wb') as temp:
            while True:
                block = pcm.read(4 * len(channels_batch) * 1048576)
                if not block: break
                data_numpy = np.frombuffer(block, dtype=np.int32).astype(np.float64) / 2**32
                freq = [data_numpy[i::len(channels_batch)] for i in range(len(channels_batch))]
                block = np.column_stack([dsd.delta_sigma(c) for c in freq]).ravel(order='C').tobytes()
                temp.write(block)
                dlen = os.path.getsize(temp_file)
                with open(temp_file, 'rb') as trd, open(dsd_name, 'wb') as dsdfile:
                    dsdfile.write(dsd.build_dff_header(dlen, channels_batch, 2822400) + trd.read())
    except KeyboardInterrupt: pass
    finally: os.remove(temp_file)

    dsd_name = 'output.dsf'
    channels = 2
    temp_file = f'temp.{base64.b64encode(secrets.token_bytes(6)).decode().replace("/", "_")}.dsd'

    # DSF #
    try:
        with open(pcm_name, 'rb') as pcm, open(temp_file, 'wb') as temp:
            while True:
                block = pcm.read(4 * channels * 1048576)
                if not block: break
                data_numpy = np.frombuffer(block, dtype=np.int32).astype(np.float64) / 2**32
                freq = [data_numpy[i::channels] for i in range(channels)]
                block = np.column_stack([dsd.delta_sigma(c) for c in freq]).ravel(order='C').tobytes()
                temp.write(block)
                dlen = os.path.getsize(temp_file)
                with open(temp_file, 'rb') as trd, open(dsd_name, 'wb') as dsdfile:
                    dsdfile.write(dsd.build_dsf_header(dlen, channels, 2822400) + trd.read())
    except KeyboardInterrupt: pass
    finally: os.remove(temp_file)
