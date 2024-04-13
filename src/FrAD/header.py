import base64, json, os, shutil, struct, sys, zlib
from .common import variables, methods
import numpy as np
from .tools.headb import headb

class header:
    def parse(file_path, output):
        try:
            open(output+'.meta.json', 'w', encoding='utf-8').write('[')
            with open(file_path, 'rb') as f:
                head = f.read(64)

                methods.signature(head[0x0:0x4])
                while True:
                    block_type = f.read(2)
                    if not block_type: break
                    if block_type == b'\xfa\xaa':
                        block_length = int.from_bytes(f.read(6), 'big')
                        title_length = int(struct.unpack('>I', f.read(4))[0])
                        title = f.read(title_length).decode('utf-8')
                        data = f.read(block_length-title_length-12)
                        try: d = {'key': title, 'type': 'string', 'value': data.decode('utf-8')}
                        except UnicodeDecodeError: d = {'key': title, 'type': 'base64', 'value': base64.b64encode(data).decode('utf-8')}
                        open(output+'.meta.json', 'a', encoding='utf-8').write(f'{json.dumps(d, ensure_ascii=False)}, ')
                    elif block_type[0] == 0xf5:
                        block_length = int(struct.unpack('>Q', f.read(8))[0])
                        open(output+'.meta.image', 'wb').write(f.read(block_length-10))
                    elif block_type == b'\xff\xd0': break
        finally:
            try:
                with open(output+'.meta.json', 'rb+') as m: m.seek(-2, 2); m.truncate(); m.write(b']')
            except: open(output+'.meta.json', 'a').write(']')

    def parse_to_ffmeta(file_path, output):
        open(output, 'w', encoding='utf-8').write(';FFMETADATA1\n')
        with open(file_path, 'rb') as f:
            head = f.read(64)

            methods.signature(head[0x0:0x4])
            while True:
                block_type = f.read(2)
                if not block_type: break
                if block_type == b'\xfa\xaa':
                    block_length = int.from_bytes(f.read(6), 'big')
                    title_length = int(struct.unpack('>I', f.read(4))[0])
                    title = f.read(title_length).decode('utf-8')
                    data = f.read(block_length-title_length-12)
                    try: d = f'{title}={data.decode('utf-8').replace('\n', '\\\n')}\n'
                    except UnicodeDecodeError: d = f'{title}={base64.b64encode(data).decode('utf-8')}\n'
                    open(output, 'a', encoding='utf-8').write(d)
                elif block_type[0] == 0xf5:
                    block_length = int(struct.unpack('>Q', f.read(8))[0])
                    open(f'{output}.image', 'wb').write(f.read(block_length-10))
                elif block_type == b'\xff\xd0': break

    def modify(file_path, meta = None, img: bytes = None):
        try:
            shutil.copy2(file_path, variables.temp)
            with open(variables.temp, 'rb') as f:
                # Fixed Header
                head = f.read(64)

                methods.signature(head[0x0:0x4])
                header_length = struct.unpack('>Q', head[0x8:0x10])[0] # 0x08-8B: Total header size

                # Backing up audio data
                f.seek(header_length)
                with open(variables.temp2, 'wb') as temp:
                    temp.write(f.read())

            # Making new header
            head_new = headb.uilder(meta, img)

            # Overwriting Fourier Analogue-in-Digital file
            with open(variables.temp, 'wb') as f: # DO NEVER DELETE THIS
                f.write(head_new)
                with open(variables.temp2, 'rb') as temp:
                    f.write(temp.read())
            shutil.move(variables.temp, file_path)
        except KeyboardInterrupt:
            print('Aborting...')
        finally:
            os.remove(variables.temp2)
            sys.exit(0)
