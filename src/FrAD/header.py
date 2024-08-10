import json, shutil, struct, sys
from .common import variables, methods, terminal
from .tools.headb import headb

class header:
    @staticmethod
    def parse(file_path, output) -> None:
        meta, img = headb.parser(file_path)
        if meta:
            open(f'{output}.meta.json', 'w', encoding='utf-8').write('[\n    ')
            meta = ',\n    '.join([json.dumps({'key': m[0], 'type': m[2], 'value': m[1]}, ensure_ascii=False) for m in meta])
            open(f'{output}.meta.json', 'a', encoding='utf-8').write(meta)
            open(f'{output}.meta.json', 'a').write('\n]')
        if img: open(f'{output}.meta.image', 'wb').write(img)

    @staticmethod
    def parse_to_ffmeta(file_path, output) -> None:
        open(output, 'w', encoding='utf-8').write(';FFMETADATA1\n')
        meta, img = headb.parser(file_path)
        for m in meta: open(output, 'a', encoding='utf-8').write(f'{m[0]}={m[1].replace('\n', '\\\n')}\n')
        if img: open(f'{output}.image', 'wb').write(img)

    @staticmethod
    def modify(file_path, meta: list | None = None, img: bytes | None = None, **kwargs) -> None:
        if file_path is None: terminal('File path is required.'); sys.exit(1)
        add = kwargs.get('add', False)
        remove = kwargs.get('remove', False)
        remove_img = kwargs.get('remove_img', False)
        try:
            # Backup file
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
                if not img: img = img_old
                if meta: meta = meta_old + meta
                else: meta = meta_old
            elif remove:
                img = img_old
                if meta: meta = [mo for mo in meta_old if mo[0] not in meta]
            elif remove_img: meta = meta_old; img = None
            head_new = headb.uilder(meta, img)

            # Overwriting Fourier Analogue-in-Digital file
            open(variables.temp, 'wb').write(head_new+open(variables.temp2, 'rb').read())
            shutil.move(variables.temp, file_path)
        except KeyboardInterrupt:
            terminal('Aborting...')
            sys.exit(0)
