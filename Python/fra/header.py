from .common import variables, methods
import math, os, shutil, struct, sys
from .tools.headb import headb

class header:
    def parse(file_path):
        d = list()
        with open(file_path, 'rb') as f:
            header = f.read(256)

            methods.signature(header[0x0:0x3])
            headlen = struct.unpack('>Q', header[0x8:0x10])[0]
            blocks = f.read(headlen - 256)
            i = 0
            image = b''
            while i < len(blocks):
                block_type = blocks[i:i+2]
                if block_type == b'\xfa\xaa':
                    block_length = int.from_bytes(blocks[i+2:i+8], 'big')
                    title_length = int(struct.unpack('>I', blocks[i+8:i+12])[0])
                    title = blocks[i+12:i+12+title_length].decode('utf-8')
                    data = blocks[i+12+title_length:i+block_length]
                    d.append([title, data])
                    i += block_length
                elif block_type[0] == 0xf5:
                    block_length = int(struct.unpack('>Q', blocks[i+2:i+10])[0])
                    data = blocks[i+10:i+block_length]
                    image = data
                    i += block_length
        return d, image

    def modify(file_path, meta = None, img: bytes = None):
        try:
            shutil.copy2(file_path, variables.temp)
            with open(variables.temp, 'rb') as f:
                # Fixed Header
                head = f.read(256)

                methods.signature(head[0x0:0x3])
                # Taking Stream info
                channel = struct.unpack('<B', head[0x3:0x4])[0] + 1    # 0x03:          Channel
                sample_rate = struct.unpack('>I', head[0x4:0x8])[0]    # 0x04-4B:       Sample rate

                header_length = struct.unpack('>Q', head[0x8:0x10])[0] # 0x08-8B:       Total header size
                checksum_stream = head[0xf0:0x100]                     # 0xf0-16B:      Stream hash

                # Backing up audio data
                f.seek(header_length)
                with open(variables.temp2, 'wb') as temp:
                    temp.write(f.read())

            # Making new header
            head_new = headb.uilder(sample_rate, channel, checksum_stream,meta, img)

            # Overwriting Fourier Analogue-in-Digital file
            with open(variables.temp, 'wb') as f: # DO NEVER DELETE THIS
                f.write(head_new)
                with open(variables.temp2, 'rb') as temp:
                    f.write(temp.read())
            os.remove(variables.temp2)
            shutil.move(variables.temp, file_path)
        except:
            os.remove(variables.temp)
            os.remove(variables.temp2)
            sys.exit(0)
