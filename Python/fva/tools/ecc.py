from multiprocessing import Pool, cpu_count
import os
from reedsolo import RSCodec, ReedSolomonError

class ecc:
    def split_data(data, chunk_size):
        return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    class rdsl:
        rs = RSCodec(16, 128)

        def encode(data):
            chunks = ecc.split_data(data, 112)
            return b''.join([bytes(ecc.rdsl.rs.encode(chunk)) for chunk in chunks])

        def decode(data):
          try:
            chunks = ecc.split_data(data, 128)
            return b''.join([bytes(ecc.rdsl.rs.decode(chunk)[0]) for chunk in chunks])
          except ReedSolomonError as e:
            print(f'Error: {e}')

    def encode(data, is_ecc_on: bool):
        if is_ecc_on == True: return ecc.rdsl.encode(data)
        if is_ecc_on == False: return data

    def decode(data, is_ecc_on: bool):
        if is_ecc_on == True: return ecc.rdsl.decode(data)
        if is_ecc_on == False: return data
