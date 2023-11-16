from reedsolo import RSCodec
import numpy as np

def split_data(data, chunk_size):
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

class cd:
    rs = RSCodec(16, 128)

    def encode(data):
        chunks = split_data(data, 112)
        return b''.join([cd.rs.encode(chunk) for chunk in chunks])

    def decode(data):
        chunks = split_data(data, 128)
        return b''.join([cd.rs.decode(chunk)[0] for chunk in chunks])

data = b''
with open('cons.wav', 'rb') as f:
    data = f.read()

encoded_data = cd.encode(data)
decoded_data = cd.decode(encoded_data)

assert data == decoded_data, "Decoded data does not match original data."
