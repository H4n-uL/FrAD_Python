from multiprocessing import Pool, cpu_count
from reedsolo import RSCodec, ReedSolomonError

class ecc:
    def split_data(data, chunk_size):
        return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

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

    def encode(data, is_ecc_on: bool):
        if is_ecc_on == True: return ecc.rdsl.encode(data)
        if is_ecc_on == False: return data

    def decode(data):
        return ecc.rdsl.decode(data)
