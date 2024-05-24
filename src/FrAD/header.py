import base64, json, os, shutil, struct, sys
from .common import variables, methods
from .tools.headb import headb

class header:
    @staticmethod
    def parse(file_path, output):
        meta, img = headb.parser(file_path)
        open(f'{output}.meta.json', 'w', encoding='utf-8').write('[')
        for m in meta: open(f'{output}.meta.json', 'a', encoding='utf-8').write(f'{json.dumps({'key': m[0], 'type': m[2], 'value': m[1]}, ensure_ascii=False)}')
        if img: open(f'{output}.meta.image', 'wb').write(img)
        try:
            with open(f'{output}.meta.json', 'rb+') as m: m.seek(-2, 2); m.truncate(); m.write(b']')
        except: open(f'{output}.meta.json', 'a').write(']')

    @staticmethod
    def parse_to_ffmeta(file_path, output):
        open(output, 'w', encoding='utf-8').write(';FFMETADATA1\n')
        meta, img = headb.parser(file_path)
        for m in meta: open(output, 'a', encoding='utf-8').write(f'{m[0]}={m[1].replace('\n', '\\\n')}\n')
        if img: open(f'{output}.image', 'wb').write(img)

    @staticmethod
    def modify(file_path, meta: list | None = None, img: bytes | None = None, **kwargs):
        if file_path is None: print('File path is required.'); sys.exit(1)
        add = kwargs.get('add', False)
        remove = kwargs.get('remove', False)
        write_img = kwargs.get('write_img', False)
        remove_img = kwargs.get('remove_img', False)
        try:
            shutil.copy2(file_path, variables.temp)
            with open(variables.temp, 'rb') as f:
                # Fixed Header
                head = f.read(64)

                if methods.signature(head[0x0:0x4]) == 'container':
                    head_len = struct.unpack('>Q', head[0x8:0x10])[0] # 0x08-8B: Total header size
                else: head_len = 0

                # Backing up audio data
                f.seek(head_len)
                with open(variables.temp2, 'wb') as temp:
                    temp.write(f.read())

            # Making new header
            meta_old, img_old = headb.parser(file_path)
            if add:
                img = img_old
                if meta: meta = meta_old + meta
            elif remove:
                img = img_old
                if meta: meta = [mo for mo in meta_old if mo[0] not in meta]
            elif write_img:
                meta = meta_old
                if img_old and not img: img = None
            elif remove_img: img = None; meta = meta_old
            head_new = headb.uilder(meta, img)

            # Overwriting Fourier Analogue-in-Digital file
            with open(variables.temp, 'wb') as f: # DO NEVER DELETE THIS
                f.write(head_new)
                with open(variables.temp2, 'rb') as temp:
                    f.write(temp.read())
            shutil.move(variables.temp, file_path)
        except KeyboardInterrupt:
            print('Aborting...')
            sys.exit(0)
