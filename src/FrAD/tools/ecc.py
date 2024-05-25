from reedsolo import RSCodec, ReedSolomonError

class ecc:
    @staticmethod
    def unecc(data: bytes, ecc_dsize: int, ecc_codesize: int) -> bytes:
        blocksize = ecc_dsize + ecc_codesize

        block = bytearray()
        for i in range(0, len(data), blocksize):
            block.extend(data[i:i+blocksize][:-ecc_codesize]) # Carrying Data Bytes from ECC chunk
        return bytes(block)

    @staticmethod
    def split_data(data: bytes, chunk_size: int):
        for i in range(0, len(data), chunk_size): yield data[i:i+chunk_size]

    @staticmethod
    def encode(data: bytes, ecc_dsize: int, ecc_codesize: int) -> bytes:
        blocksize = ecc_dsize + ecc_codesize
        rs = RSCodec(ecc_codesize, blocksize)

        encoded_chunks = [bytes(rs.encode(chunk)) for chunk in ecc.split_data(data, ecc_dsize)]
        data = b''.join(encoded_chunks)
        return data

    @staticmethod
    def decode(data: bytes, ecc_dsize: int, ecc_codesize: int) -> bytes:
        blocksize = ecc_dsize + ecc_codesize
        rs = RSCodec(ecc_codesize, blocksize)

        decoded_chunk = []
        for chunk in ecc.split_data(data, blocksize):
            try:
                decoded_chunk.append(bytes(rs.decode(chunk)[0]))
            except ReedSolomonError as e:
                decoded_chunk.append(b'\x00'*ecc_dsize)
        return b''.join(decoded_chunk)
