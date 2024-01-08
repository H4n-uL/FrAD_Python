from ..common import variables
from multiprocessing import Pool, cpu_count
from reedsolo import RSCodec, ReedSolomonError

rs = RSCodec(variables.ecc.code_size, variables.ecc.block_size)

class ecc:
    def unecc(data):
        block = bytearray()
        for i in range(0, len(data), variables.ecc.block_size):
            block.extend(data[i:i+variables.ecc.block_size][:-variables.ecc.code_size]) # Carrying first 128 Bytes data from 148 Bytes chunk
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
            chunks = ecc.split_data(data, variables.ecc.data_size)
            with Pool(cpu_count() // 2) as p:
                encoded_chunks = p.map(ecc.rdsl.encode_chunk, chunks)
            return b''.join(encoded_chunks)

        def decode(data):
            chunks = ecc.split_data(data, variables.ecc.block_size)
            with Pool(cpu_count() // 2) as p:
                decoded_chunks = p.map(ecc.rdsl.decode_chunk, chunks)
            return b''.join(decoded_chunks)

    def encode(data):
        return ecc.rdsl.encode(data)

    def decode(data):
        return ecc.rdsl.decode(data)
