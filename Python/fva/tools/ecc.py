from multiprocessing import Pool, cpu_count
import os
from reedsolo import RSCodec, ReedSolomonError

class ecc:
    ENCODE_OPTIONS = {
        None: 0b000,
        'none': 0b000,
        'hamming': 0b001, 'hamm': 0b001,
        'golay': 0b010,
        'convolutional': 0b011, 'conv': 0b011,
        'reedsolomon': 0b100, 'rdsl': 0b100,
        'bch': 0b101,
        'lpdc': 0b110,
        'turbo': 0b111,
        0b000: 0b000,
        0b001: 0b001,
        0b010: 0b010,
        0b011: 0b011,
        0b010: 0b100,
        0b101: 0b101,
        0b110: 0b110,
        0b11: 0b111,
        0: 0b000,
        1: 0b001,
        2: 0b010,
        3: 0b011,
        4: 0b100,
        5: 0b101,
        6: 0b110,
        7: 0b111
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
        if ecc.ENCODE_OPTIONS[option] == 0b000: return data
        elif ecc.ENCODE_OPTIONS[option] == 0b100: return ecc.rdsl.encode(data)
        else:
            print('Not supported yet')
            return data

    def decode(data_rs, option):
        if option == 0b000: return data_rs
        elif option == 0b100: return ecc.rdsl.decode(data_rs)
        else:
            print('Not supported yet')
            return data_rs
