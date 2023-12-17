import struct

class cb:
    IMAGE =   b'\xf5\x55'
    COMMENT = b'\xfa\xaa'

    def comment(title, data):
        if type(data) == str: data = data.encode('utf-8')
        title_length = struct.pack('<I', len(title))
        data_comb = title.encode('utf-8') + data
        block_length = (len(data_comb) + 12).to_bytes(6, 'little')
        _block = cb.COMMENT + block_length + title_length + data_comb
        return _block

    def image(data):
        block_length = struct.pack('<Q', len(data) + 10)
        _block = cb.IMAGE + block_length + data
        return _block
