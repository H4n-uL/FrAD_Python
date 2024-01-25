from ..common import variables, ecc_v
from multiprocessing import Pool, cpu_count
from reedsolo import RSCodec, ReedSolomonError

class ECCUnavailable(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

if ecc_v.data_size is None:
    raise ECCUnavailable(f'ECC unavailable for this block size: {variables.nperseg}')

rs = RSCodec(ecc_v.code_size, ecc_v.block_size)

class ecc:
    def unecc(data):
        block = bytearray()
        for i in range(0, len(data), ecc_v.block_size):
            block.extend(data[i:i+ecc_v.block_size][:-ecc_v.code_size]) # Carrying first Data Bytes data from ECC chunk
        return bytes(block)

    def split_data(data, chunk_size):
        for i in range(0, len(data), chunk_size): yield data[i:i+chunk_size]

    class rdsl:
        def encode_chunk(chunk):
            return bytes(rs.encode(chunk))

        def decode_chunk(chunk):
            try:
                return bytes(rs.decode(chunk)[0])
            except ReedSolomonError as e:
                print(f'Error: {e}')
                return None

        def encode(data):
            chunks = ecc.split_data(data, ecc_v.data_size)
            with Pool(cpu_count() // 2) as p:
                encoded_chunks = p.map(ecc.rdsl.encode_chunk, chunks)
            return b''.join(encoded_chunks)

        def decode(data):
            chunks = ecc.split_data(data, ecc_v.block_size)
            with Pool(cpu_count() // 2) as p:
                decoded_chunks = p.map(ecc.rdsl.decode_chunk, chunks)
            return b''.join(decoded_chunks)

    def encode(data):
        return ecc.rdsl.encode(data)

    def decode(data):
        return ecc.rdsl.decode(data)
