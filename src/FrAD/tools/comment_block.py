import struct

class cb:
    IMAGE =   b'\xf5'
    COMMENT = b'\xfa\xaa'

    def comment(title, data):
        if type(data) == str: data = data.encode('utf-8')
        title_length = struct.pack('>I', len(title))
        data_comb = title.encode('utf-8') + data
        block_length = (len(data_comb) + 12).to_bytes(6, 'big')
        _block = cb.COMMENT + block_length + title_length + data_comb
        return _block

    def image(data, pictype = 3):
        apictype = struct.pack('<B', 0b01000000 | pictype)
        block_length = struct.pack('>Q', len(data) + 10)
        _block = cb.IMAGE + apictype + block_length + data
        return _block
