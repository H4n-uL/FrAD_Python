import struct

block_types = {
    b'T': 'TITLE',
    b'R': 'LIRYCS',
    b'A': 'ARTIST',
    b'a': 'ALBUM',
    b't': 'TRACKNUMBER',
    b'G': 'GENRE',
    b'D': 'DATE',
    b'd': 'DESCRIPTION',
    b'C': 'COMMENT',
    b'M': 'COMPOSER',
    b'c': 'COPYRIGHT',
    b'L': 'LICENSE',
    b'O': 'ORGANIZATION',
    b'l': 'LOCATION',
    b'P': 'PERFORMER',
    b'I': 'ISRC',
    b'i': 'IMAGE'
}

b3_to_bits = {
    0b110: 512,
    0b101: 256,
    0b100: 128,
    0b011: 64,
    0b010: 32,
    0b001: 16
}

ecc_options = {
    0b001: 'digitalfile',
    0b010: 'reedsolomon',
    0b011: 'advancedformat'
}

class parser:
    def parse(file_path):
        d = dict()
        with open(file_path, 'rb') as f:
            header = f.read(256)

            d['signature'] = header[0x0:0xa]
            d['headlen'] = struct.unpack('<Q', header[0xa:0x12])[0]
            d['samprate'] = int.from_bytes(header[0x12:0x15], 'little')
            cfb = struct.unpack('<B', header[0x15:0x16])[0]
            d['channel'] = cfb >> 3
            d['bitrate'] = b3_to_bits.get(cfb & 0b111, None)
            d['ecc'] = ecc_options.get(struct.unpack('<B', header[0x16:0x17])[0] >> 5)

            blocks = f.read(d['headlen'] - 256)
            i = 0
            while i < len(blocks):
                block_type = blocks[i:i+1]
                block_length = int(struct.unpack('<I', blocks[i+1:i+5])[0])
                block_data = blocks[i+5:i+block_length].decode('utf-8')
                d[f'{block_types.get(block_type, "UNKNOWN")}'] = block_data
                i += block_length

        return d
