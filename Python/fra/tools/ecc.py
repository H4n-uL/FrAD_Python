from reedsolo import RSCodec, ReedSolomonError

class ecc:
    def unecc(data, ecc_dsize, ecc_codesize):
        blocksize = ecc_dsize + ecc_codesize

        block = bytearray()
        for i in range(0, len(data), blocksize):
            block.extend(data[i:i+blocksize][:-ecc_codesize]) # Carrying Data Bytes from ECC chunk
        return bytes(block)

    def split_data(data, chunk_size):
        for i in range(0, len(data), chunk_size): yield data[i:i+chunk_size]

    def encode(data, ecc_dsize, ecc_codesize):
        blocksize = ecc_dsize + ecc_codesize
        rs = RSCodec(ecc_codesize, blocksize)

        chunks = ecc.split_data(data, ecc_dsize)
        encoded_chunks = [bytes(rs.encode(chunk)) for chunk in chunks]
        data = b''.join(encoded_chunks)
        return data

    def decode(data, ecc_dsize, ecc_codesize):
        blocksize = ecc_dsize + ecc_codesize
        rs = RSCodec(ecc_codesize, blocksize)

        chunks = ecc.split_data(data, blocksize)
        try: decoded_chunks = [bytes(rs.decode(chunk)[0]) for chunk in chunks]
        except ReedSolomonError as e: print(f'Error: {e}'); return None
        return b''.join(decoded_chunks)
