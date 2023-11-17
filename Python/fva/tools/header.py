from .comment_block import cb
from .ecc import ecc as ecc_class
import struct

bits_to_b3 = {
    512: 0b110,
    256: 0b101,
    128: 0b100,
    64: 0b011,
    32: 0b010,
    16: 0b001
}

class header:
    def builder(
            sample_rate_bytes: bytes, channel: int, bits: int, ecc: str = None,
            title: str = None, lyrics: str = None, artist: str = None,
            album: str = None, track_number: int = None, genre: str = None,
            date: str = None, description: str = None, comment: str = None,
            composer: str = None, copyright: str = None, license: str = None,
            organization: str = None, location: str = None, performer: str = None,
            isrc: str = None, img: bytes = None):
        b3 = bits_to_b3.get(bits, 0b000)

        cfb = (channel << 3) | b3

        signature = b'\x7e\x8b\xab\x89\xea\xc0\x9d\xa9\x68\x80'
        length = b'\x00'*8; sample_rate_bytes
        cfb_struct = struct.pack('<B', cfb)
        ecc_bits = struct.pack('<B', ecc_class.ENCODE_OPTIONS[ecc] << 5 | 0b00000)
        reserved = b'\x00'*233

        blocks = bytes()

        if title is not None: blocks += cb.header_block_gen(title, cb.TITLE)
        if lyrics is not None: blocks += cb.header_block_gen(lyrics, cb.LYRICS)
        if artist is not None: blocks += cb.header_block_gen(artist, cb.ARTIST)
        if album is not None: blocks += cb.header_block_gen(album, cb.ALBUM)
        if track_number is not None: blocks += cb.header_block_gen(track_number, cb.TRACKNUMBER)
        if genre is not None: blocks += cb.header_block_gen(genre, cb.GENRE)
        if date is not None: blocks += cb.header_block_gen(date, cb.DATE)
        if description is not None: blocks += cb.header_block_gen(description, cb.DESCRIPTION)
        if comment is not None: blocks += cb.header_block_gen(comment, cb.COMMENT)
        if composer is not None: blocks += cb.header_block_gen(composer, cb.COMPOSER)
        if copyright is not None: blocks += cb.header_block_gen(copyright, cb.COPYRIGHT)
        if license is not None: blocks += cb.header_block_gen(license, cb.LICENSE)
        if organization is not None: blocks += cb.header_block_gen(organization, cb.ORGANIZATION)
        if location is not None: blocks += cb.header_block_gen(location, cb.LOCATION)
        if performer is not None: blocks += cb.header_block_gen(performer, cb.PERFORMER)
        if isrc is not None: blocks += cb.header_block_gen(isrc, cb.ISRC)
        if img is not None: blocks += cb.header_block_gen(img, cb.IMAGE)

        length = struct.pack('<Q', (len(signature + length + sample_rate_bytes + cfb_struct + ecc_bits + reserved + blocks)))

        header = signature + length + sample_rate_bytes + cfb_struct + ecc_bits + reserved + blocks
        return header
