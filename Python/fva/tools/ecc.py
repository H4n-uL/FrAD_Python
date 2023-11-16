from multiprocessing import Pool, cpu_count
import os
from reedsolo import RSCodec, ReedSolomonError

class ecc:
    ENCODE_OPTIONS = {
        None: [None, None, 0b000],
        'digi': [8, 128, 0b001],
        'cd': [288, 2336, 0b010],
        'af': [820, 4096, 0b011],
        'digitalfile': [8, 128, 0b001],
        'cd-digitalaudio': [288, 2336, 0b010],
        'advancedformat': [820, 4096, 0b011],
        0b001: [8, 128, 0b001],
        0b010: [288, 2336, 0b010],
        0b011: [820, 4096, 0b011],
        1: [8, 128, 0b001],
        2: [288, 2336, 0b010],
        3: [820, 4096, 0b011]
    }

    DECODE_OPTIONS = {
        0b000: [None, None, 0b000],
        0b001: [8, 128, 0b001],
        0b010: [288, 2336, 0b010],
        0b011: [820, 4096, 0b011]
    }

    def split_data(data, chunk_size):
        return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    class cd:
        rs = RSCodec(16, 128)

        def encode(data):
            chunks = ecc.split_data(data, 112)
            return b''.join([ecc.cd.rs.encode(chunk) for chunk in chunks])

        def decode(data):
            chunks = ecc.split_data(data, 128)
            return b''.join([ecc.cd.rs.decode(chunk)[0] for chunk in chunks])

    def encode(data, option):
        if option is None: return data
        elif ecc.ENCODE_OPTIONS[option][2] == 0b010: return ecc.cd.encode(data)
        else:
            print('Not supported yet')
            return data

    def decode(data_rs, option):
        if option == 0b000: return data_rs
        elif option == 0b010: return ecc.cd.decode(data_rs)
        else:
            print('Not supported yet')
            return data_rs
