from multiprocessing import Pool, cpu_count
import os
from reedsolo import RSCodec, ReedSolomonError

class ecc:
    ENCODE_OPTIONS = {
        None: [None, None, 0b000],
        'digi': [16, 256, 0b001],
        'cd': [288, 2336, 0b010],
        'af': [820, 4096, 0b011],
        'digitalfile': [16, 256, 0b001],
        'cd-digitalaudio': [288, 2336, 0b010],
        'advancedformat': [820, 4096, 0b011],
        0b001: [16, 256, 0b001],
        0b010: [288, 2336, 0b010],
        0b011: [820, 4096, 0b011],
        1: [16, 256, 0b001],
        2: [288, 2336, 0b010],
        3: [820, 4096, 0b011]
    }

    DECODE_OPTIONS = {
        0b000: [None, None, 0b000],
        0b001: [16, 256, 0b001],
        0b010: [288, 2336, 0b010],
        0b011: [820, 4096, 0b011]
    }

    # def encode_block(args):
    #     rs, block, i, total = args
    #     print(f'\033[A\033[K{i} / {total}')
    #     encoded = rs.encode(block.tobytes())
    #     return encoded

    # def encode(data, option):
    #     if option is None: return data.tobytes()

    #     rs = RSCodec(ecc.ENCODE_OPTIONS[option][0], ecc.ENCODE_OPTIONS[option][1])
    #     block_size = ecc.ENCODE_OPTIONS[option][1] - ecc.ENCODE_OPTIONS[option][0]

    #     # CPU 코어 개수가 4개 이상인 경우에만 멀티프로세싱을 적용합니다.
    #     num_processes = cpu_count() // 2 if cpu_count() >= 4 else 1

    #     # 데이터를 프로세스 수만큼 분할합니다. 단, 각 청크의 크기가 block_size의 배수가 되도록 조절합니다.
    #     data_len = len(data)
    #     chunk_size = (data_len // num_processes) // block_size * block_size
    #     data_chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]
    #     if data_len % num_processes != 0:
    #         remainder = data[num_processes*chunk_size:]
    #         remainder_size = (len(remainder) // block_size) * block_size
    #         data_chunks[-1] += remainder[:remainder_size]

    #     # 각 데이터 청크를 block_size 단위로 분할합니다.
    #     blocks = [(rs, data_chunk[i*block_size:(i+1)*block_size], i, num_processes) for data_chunk in data_chunks for i in range(len(data_chunk) // block_size)]

    #     print()

    #     with Pool(num_processes) as p:
    #         data_rs = bytearray().join(p.imap(ecc.encode_block, blocks))

    #     return data_rs

    def encode(data, option):
        if option is None: return data

        rs = RSCodec(ecc.ENCODE_OPTIONS[option][0], ecc.ENCODE_OPTIONS[option][1])
        block_size = ecc.ENCODE_OPTIONS[option][1] - ecc.ENCODE_OPTIONS[option][0]

        num_blocks = len(data) // block_size
        data_rs = bytearray()

        for i in range(num_blocks):
            print(num_blocks,'/',i)
            block = data[i*block_size:(i+1)*block_size]
            data_rs.extend(rs.encode(block.tobytes()))
            print("\033[A\033[K", end="")

        if len(data) % block_size != 0:
            block = data[num_blocks*block_size:]
            data_rs.extend(rs.encode(block.tobytes()))

        return data_rs

    def decode(data_rs, option):
        if option is None: return data_rs.tobytes()

        rs = RSCodec(ecc.DECODE_OPTIONS[option][0], ecc.DECODE_OPTIONS[option][1])
        block_size = ecc.DECODE_OPTIONS[option][1]

        num_blocks = len(data_rs) // block_size
        data = bytearray()

        for i in range(num_blocks):
            print(num_blocks,'/',i)
            block = data_rs[i*block_size:(i+1)*block_size]
            try:
                data.extend(rs.decode(block))
            except ReedSolomonError as e:
                print(f"\033[A\033[KBlock {i} failed to decode: {e}")
                print()
                # Handle error as needed
            print("\033[A\033[K", end="")

        if len(data_rs) % block_size != 0:
            block = data_rs[num_blocks*block_size:]
            try:
                data.extend(rs.decode(block))
            except ReedSolomonError as e:
                print(f"Last block failed to decode: {e}")
                # Handle error as needed

        return data
