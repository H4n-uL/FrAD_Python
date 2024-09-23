from libfrad.common import SIGNATURE

COMMENT = b'\xfa\xaa'
IMAGE = b'\xf5'

def comment(title: str, data: bytes) -> bytes:
    title_length = len(title.encode()).to_bytes(4, 'big')
    data_comb = title.encode() + data
    block_length = (len(data_comb) + 12).to_bytes(6, 'big')
    return bytes(COMMENT + block_length + title_length + data_comb)

def image(data: bytes, pictype: int | None = None) -> bytes:
    pictype = pictype or 3
    pictype = 3 if pictype > 20 else pictype
    apictype = bytes([0b01000000 | pictype])
    block_length = (len(data) + 10).to_bytes(8, 'big')
    return bytes(IMAGE + apictype + block_length + data)

def builder(meta: list[tuple[str, bytes]], img: bytes) -> bytes:
    blocks = b''

    if meta:
        for title, data in meta:
            blocks += comment(title, data)
    if img:
        blocks += image(img)

    length = (64 + len(blocks)).to_bytes(8, 'big')

    header = bytes(
        SIGNATURE +
        b'\x00\x00\x00\x00' +
        length +
        b'\x00' * 48 +
        blocks
    )

    return header

def parser(header: bytes) -> tuple[list[tuple[str, bytes]], bytes]:
    meta = []
    img = b''

    while len(header) >= 2:
        block_type = header[:2]
        if block_type == COMMENT:
            block_length = int.from_bytes(b'\x00\x00' + header[2:8], 'big')
            title_length = int.from_bytes(header[8:12], 'big')

            title = header[12:12 + title_length].decode()
            data = header[12 + title_length:block_length]
            meta.append((title, data))
            header = header[block_length:]
        elif block_type[0] == IMAGE[0]:
            block_length = int.from_bytes(header[2:10], 'big')
            img = header[10:block_length]
            header = header[block_length:]
        else:
            header = header[1:]

    return meta, img
