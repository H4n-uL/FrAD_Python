from reedsolo import RSCodec, ReedSolomonError
import numpy as np

def split_data(data, chunk_size):
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

class cd:
    rs = RSCodec(16, 128)

    def encode(data):
        chunks = split_data(data, 112)
        enc = b''
        for i, chunk in enumerate(chunks):
            print(i,'/',len(chunks))
            enc += cd.rs.encode(chunk)
            print('\033[A\033[K', end='')
        return enc

    def decode(data):
        try:
            chunks = split_data(data, 128)
            dec = b''
            for i, chunk in enumerate(chunks):
                print(i,'/',len(chunks))
                dec += cd.rs.decode(chunk)[0]
                print('\033[A\033[K', end='')
            return dec
        except ReedSolomonError as e:
            print('Error:', e)

data = b''
with open('cons.wav', 'rb') as f:
    data = f.read()
# data = np.random.bytes(2**23)
encoded_data = cd.encode(data)
decoded_data = cd.decode(encoded_data)
# print(decoded_data)

assert data == decoded_data, "Decoded data does not match original data."
