import struct

class cb:
    TITLE =        b'T'
    LYRICS =       b'R'
    ARTIST =       b'A'
    ALBUM =        b'a'
    TRACKNUMBER =  b't'
    GENRE =        b'G'
    DATE =         b'D'
    DESCRIPTION =  b'd'
    COMMENT =      b'C'
    COMPOSER =     b'M'
    COPYRIGHT =    b'c'
    LICENSE =      b'L'
    ORGANIZATION = b'O'
    LOCATION =     b'l'
    PERFORMER =    b'P'
    ISRC =         b'I'
    IMAGE =        b'i'

    def header_block_gen(data, type):
        block_type = type
        block_data = data.encode('utf-8')
        block_length = struct.pack('<I', len(block_data + block_type) + 4)

        _block = block_type + block_length + block_data
        return _block
