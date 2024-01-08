from multiprocessing import Pool, cpu_count
from reedsolo import RSCodec, ReedSolomonError

class ecc:
    def unecc(data):
        blocksize, dsize = 148, 128
        block = bytearray()
        for i in range(0, len(data), blocksize):
            block.extend(data[i:i+dsize]) # Carrying first 128 Bytes data from 148 Bytes chunk
        return bytes(block)

    def split_data(data, chunk_size):
        for i in range(0, len(data), chunk_size): yield data[i:i+chunk_size]

    class rdsl:
        rs = RSCodec(20, 148)

        def encode_chunk(chunk):
            return bytes(ecc.rdsl.rs.encode(chunk))

        def decode_chunk(chunk):
            try:
                return bytes(ecc.rdsl.rs.decode(chunk)[0])
            except ReedSolomonError as e:
                print(f'Error: {e}')
                return None

        def encode(data):
            chunks = ecc.split_data(data, 128)
            with Pool(cpu_count() // 2) as p:
                encoded_chunks = p.map(ecc.rdsl.encode_chunk, chunks)
            return b''.join(encoded_chunks)

        def decode(data):
            chunks = ecc.split_data(data, 148)
            with Pool(cpu_count() // 2) as p:
                decoded_chunks = p.map(ecc.rdsl.decode_chunk, chunks)
            return b''.join(decoded_chunks)

    def encode(data):
        return ecc.rdsl.encode(data)

    def decode(data):
        return ecc.rdsl.decode(data)
