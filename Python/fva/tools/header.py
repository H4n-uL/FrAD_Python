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
            # Fixed Header
            sample_rate_bytes: bytes, channel: int,
            bits: int, isecc: bool, md5: bytes,

            # Metadata
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
        isecc = (0b1 if isecc else 0b0) << 7
        ecc_bits = struct.pack('<B', isecc | 0b0000000)
        reserved = b'\x00'*217

        blocks = bytes()

        if title is not None: blocks += cb.typical(title, cb.TITLE)
        if lyrics is not None: blocks += cb.typical(lyrics, cb.LYRICS)
        if artist is not None: blocks += cb.typical(artist, cb.ARTIST)
        if album is not None: blocks += cb.typical(album, cb.ALBUM)
        if track_number is not None: blocks += cb.typical(track_number, cb.TRACKNUMBER)
        if genre is not None: blocks += cb.typical(genre, cb.GENRE)
        if date is not None: blocks += cb.typical(date, cb.DATE)
        if description is not None: blocks += cb.typical(description, cb.DESCRIPTION)
        if comment is not None: blocks += cb.typical(comment, cb.COMMENT)
        if composer is not None: blocks += cb.typical(composer, cb.COMPOSER)
        if copyright is not None: blocks += cb.typical(copyright, cb.COPYRIGHT)
        if license is not None: blocks += cb.typical(license, cb.LICENSE)
        if organization is not None: blocks += cb.typical(organization, cb.ORGANIZATION)
        if location is not None: blocks += cb.typical(location, cb.LOCATION)
        if performer is not None: blocks += cb.typical(performer, cb.PERFORMER)
        if isrc is not None: blocks += cb.typical(isrc, cb.ISRC)
        if img is not None: blocks += cb.typical(img, cb.IMAGE)

        length = struct.pack('<Q', (len(signature + length + sample_rate_bytes + cfb_struct + ecc_bits + reserved + md5 + blocks)))

        header = signature + length + sample_rate_bytes + cfb_struct + ecc_bits + reserved + md5 + blocks
        return header
