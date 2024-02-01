import struct

with open('tmp.frad', 'rb') as f:
    header = f.read(64)
    f.seek(struct.unpack('>Q', header[0x8:0x10])[0])
    data = f.read()

with open('appended.fra', 'ab') as f:
    f.write(data)