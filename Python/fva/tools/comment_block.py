import struct

class cb:
    TITLE =        b'\xb6\x57'
    LYRICS =       b'\x97\x2a'
    ARTIST =       b'\x6a\xbb'
    ALBUM =        b'\x6a\x56'
    TRACKNUMBER =  b'\xb6\xb9'
    GENRE =        b'\x82\x7a'
    DATE =         b'\x75\xab'
    DESCRIPTION =  b'\x75\xeb'
    COMMENT =      b'\x72\x89'
    COMPOSER =     b'\x72\x6a'
    COPYRIGHT =    b'\x72\x9a'
    LICENSE =      b'\x96\x27'
    ORGANIZATION = b'\xa2\xb8'
    LOCATION =     b'\x96\x87'
    PERFORMER =    b'\xa6\xb7'
    ISRC =         b'\x8a\xca'
    IMAGE =        b'\x8a\x68'
    CUSTOM =       b'\x72\xeb'

    def typical(data, type):
        block_type = type
        block_data = data.encode('utf-8')
        block_length = struct.pack('<I', len(block_data) + 5)

        _block = block_type + block_length + block_data
        return _block

    def custom(title, data):
        block_type = cb.CUSTOM
        title_length = struct.pack('<I', len(title))
        data_comb = title + data
        block_length = (len(data_comb) + 12).to_bytes(6, 'little')

        _block = block_type + block_length + title_length + data_comb
        return _block
