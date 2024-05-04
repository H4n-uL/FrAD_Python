from reedsolo import RSCodec, ReedSolomonError

class ecc:
    @staticmethod
    def unecc(data, ecc_dsize, ecc_codesize):
        blocksize = ecc_dsize + ecc_codesize

        block = bytearray()
        for i in range(0, len(data), blocksize):
            block.extend(data[i:i+blocksize][:-ecc_codesize]) # Carrying Data Bytes from ECC chunk
        return bytes(block)

    @staticmethod
    def split_data(data, chunk_size):
        for i in range(0, len(data), chunk_size): yield data[i:i+chunk_size]

    @staticmethod
    def encode(data, ecc_dsize, ecc_codesize):
        blocksize = ecc_dsize + ecc_codesize
        rs = RSCodec(ecc_codesize, blocksize)

        chunks = ecc.split_data(data, ecc_dsize)
        encoded_chunks = [bytes(rs.encode(chunk)) for chunk in chunks]
        data = b''.join(encoded_chunks)
        return data

    @staticmethod
    def decode(data, ecc_dsize, ecc_codesize):
        blocksize = ecc_dsize + ecc_codesize
        rs = RSCodec(ecc_codesize, blocksize)

        chunks = ecc.split_data(data, blocksize)
        decoded_chunk = []
        for chunk in chunks:
            try:
                decoded_chunk.append(bytes(rs.decode(chunk)[0]))
            except ReedSolomonError as e:
                decoded_chunk.append(b'\x00'*ecc_dsize)
        return b''.join(decoded_chunk)
