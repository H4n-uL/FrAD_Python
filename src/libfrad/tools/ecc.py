from reedsolo import RSCodec, ReedSolomonError

def unecc(data: bytes, ecc_dsize: int, ecc_codesize: int) -> bytes:
    blocksize = ecc_dsize + ecc_codesize

    block = bytearray()
    for i in range(0, len(data), blocksize):
        block.extend(data[i:i+blocksize][:-ecc_codesize])
    return bytes(block)

def split_data(data: bytes, chunk_size: int):
    for i in range(0, len(data), chunk_size): yield data[i:i+chunk_size]

def encode(data: bytes, ecc_dsize: int, ecc_codesize: int) -> bytes:
    blocksize = ecc_dsize + ecc_codesize
    rs = RSCodec(ecc_codesize, blocksize)

    encoded_chunks = [bytes(rs.encode(chunk)) for chunk in split_data(data, ecc_dsize)]
    data = b''.join(encoded_chunks)
    return data

def decode(data: bytes, ecc_dsize: int, ecc_codesize: int, repair: bool) -> bytes:
    if repair:
        blocksize = ecc_dsize + ecc_codesize
        rs = RSCodec(ecc_codesize, blocksize)

        decoded_chunk = []
        for chunk in split_data(data, blocksize):
            try: decoded_chunk.append(bytes(rs.decode(chunk)[0]))
            except ReedSolomonError as _: decoded_chunk.append(b'\x00'*ecc_dsize)

        return b''.join(decoded_chunk)
    else:
        return unecc(data, ecc_dsize, ecc_codesize)
