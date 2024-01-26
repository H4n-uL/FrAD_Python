from ..common import variables, ecc_v
from multiprocessing import Pool, cpu_count
from reedsolo import RSCodec, ReedSolomonError

class ECCUnavailable(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

rs = RSCodec(ecc_v.code_size, ecc_v.block_size)

class ecc:
    def unecc(data):
        block = bytearray()
        for i in range(0, len(data), ecc_v.block_size):
            block.extend(data[i:i+ecc_v.block_size][:-ecc_v.code_size]) # Carrying first Data Bytes data from ECC chunk
        return bytes(block)

    def split_data(data, chunk_size):
        for i in range(0, len(data), chunk_size): yield data[i:i+chunk_size]

    def encode(data):
        chunks = ecc.split_data(data, ecc_v.data_size)
        encoded_chunks = [bytes(rs.encode(chunk)) for chunk in chunks]
        return b''.join(encoded_chunks)

    def decode(data):
        chunks = ecc.split_data(data, ecc_v.block_size)
        try: decoded_chunks = [bytes(rs.decode(chunk)[0]) for chunk in chunks]
        except ReedSolomonError as e: print(f'Error: {e}'); return None
        return b''.join(decoded_chunks)
