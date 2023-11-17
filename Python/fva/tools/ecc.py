from multiprocessing import Pool, cpu_count
import os
from reedsolo import RSCodec, ReedSolomonError

class ecc:
    ENCODE_OPTIONS = {
        None: 0b000,
        'digi': 0b001,
        'rdsl': 0b010,
        'af': 0b011,
        'digitalfile': 0b001,
        'reedsolomon': 0b010,
        'advancedformat': 0b011,
        0b001: 0b001,
        0b010: 0b010,
        0b011: 0b011,
        1: 0b001,
        2: 0b010,
        3: 0b011
    }

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

    def encode(data, option):
        if option is None: return data
        elif ecc.ENCODE_OPTIONS[option] == 0b010: return ecc.rdsl.encode(data)
        else:
            print('Not supported yet')
            return data

    def decode(data_rs, option):
        if option == 0b000: return data_rs
        elif option == 0b010: return ecc.rdsl.decode(data_rs)
        else:
            print('Not supported yet')
            return data_rs
