import base64, json, os, shutil, struct, sys, zlib
from .common import variables, methods
import numpy as np
from .tools.headb import headb

class header:
    # def signal(data: bytes, fb: int, channels: int, little_endian: bool):
    #     endian = not little_endian and '>' or '<'
    #     dt = {0b101:'d',0b100:'d',0b011:'f',0b010:'f',0b001:'e',0b000:'e'}[fb]
    #     if fb in [0b101,0b011,0b001]:
    #         pass
    #     elif fb in [0b100,0b010]:
    #         if fb == 0b100: data = b''.join([not little_endian and data[i:i+6]+b'\x00\x00' or b'\x00\x00'+data[i:i+6] for i in range(0, len(data), 6)])
    #         elif fb == 0b010: data = b''.join([not little_endian and data[i:i+3]+b'\x00' or b'\x00'+data[i:i+3] for i in range(0, len(data), 3)])
    #     elif fb == 0b000:
    #         data = data.hex()
    #         if endian == '<': data = ''.join([data[i:i+3][0] + '0' + data[i:i+3][1:] for i in range(0, len(data), 3)])
    #         else: data = ''.join([data[i:i+3] + '0' for i in range(0, len(data), 3)])
    #         data = bytes.fromhex(data)
    #     else:
    #         raise Exception('Illegal bits value.')
    #     freq = np.frombuffer(data, dtype=endian+dt).astype(float)
    #     freq = np.where(np.isnan(freq) | np.isinf(freq), 0, freq)
    #     freq = {f'bin {i}': list(freq[i:i+channels]) for i in range(len(freq)//channels)}
    #     return freq

    # def parse_all(file_path, output):
    #     try:
    #         open(output+'.audio.json', 'w', encoding='utf-8').write('[')
    #         open(output+'.meta.json', 'w', encoding='utf-8').write('[')
    #         with open(file_path, 'rb') as f:
    #             head = f.read(64)

    #             methods.signature(head[0x0:0x4])
    #             image = b''
    #             while True:
    #                 block_type = f.read(2)
    #                 if not block_type: break
    #                 if block_type == b'\xfa\xaa':
    #                     block_length = int.from_bytes(f.read(6), 'big')
    #                     title_length = int(struct.unpack('>I', f.read(4))[0])
    #                     title = f.read(title_length).decode('utf-8')
    #                     data = f.read(block_length-title_length-12)
    #                     try: d = {'key': title, 'type': 'string', 'value': data.decode('utf-8')}
    #                     except UnicodeDecodeError: d = {'key': title, 'type': 'base64', 'value': base64.b64encode(data).decode('utf-8')}
    #                     open(output+'.meta.json', 'a', encoding='utf-8').write(json.dumps(d, ensure_ascii=False))
    #                     open(output+'.meta.json', 'a', encoding='utf-8').write(', ')

    #                 elif block_type[0] == 0xf5:
    #                     block_length = int(struct.unpack('>Q', f.read(8))[0])
    #                     open(output+'.meta.image', 'wb').write(f.read(block_length-10))

    #                 elif block_type == b'\xff\xd0':
    #                     f.read(2)
    #                     fheadinfo = f.read(28)
    #                     framelength = struct.unpack('>I', fheadinfo[0x0:0x4])[0]        # 0x04-4B: Audio Stream Frame length
    #                     efb = struct.unpack('>B', fheadinfo[0x4:0x5])[0]                # 0x08:    Cosine-Float Bit
    #                     lossy, is_ecc_on, endian, float_bits = headb.decode_efb(efb)
    #                     channels_frame = struct.unpack('>B', fheadinfo[0x5:0x6])[0] + 1 # 0x09:    Channels
    #                     ecc_dsize = struct.unpack('>B', fheadinfo[0x6:0x7])[0]          # 0x0a:    ECC Data block size
    #                     ecc_codesize = struct.unpack('>B', fheadinfo[0x7:0x8])[0]       # 0x0b:    ECC Code size
    #                     srate_frame = struct.unpack('>I', fheadinfo[0x8:0xc])[0]        # 0x0c-4B: Sample rate
    #                     crc32 = fheadinfo[0x18:0x1c]                                    # 0x1c-4B: ISO 3309 CRC32 of Audio Data
    #                     frame = f.read(framelength)
    #                     if lossy: frame = zlib.decompress(frame)

    #                     audio = {
    #                         'block_header': {
    #                             'sync': '1111 1111 1101 0000 1101 0010 1001 0111',
    #                             'frm_length': framelength,
    #                             'efb': {
    #                                 'lossy_compression': lossy,
    #                                 'ecc': is_ecc_on,
    #                                 'endian': endian and 'big' or 'little',
    #                                 'bit_depth': {0b110:128,0b101:64,0b100:48,0b011:32,0b010:24,0b001:16,0b000:12}[float_bits]
    #                             },
    #                             'channels': channels_frame,
    #                             'ecc-data_ratio': [ecc_dsize, ecc_codesize],
    #                             'sample_rate': srate_frame,
    #                             'crc32': crc32.hex(),
    #                         },
    #                         'data': header.signal(frame, float_bits, channels_frame, endian)
    #                     }
    #                     open(output+'.audio.json', 'a', encoding='utf-8').write(json.dumps(audio, ensure_ascii=False))
    #                     open(output+'.audio.json', 'a', encoding='utf-8').write(', ')
    #     finally:
    #         try:
    #             with open(output+'.meta.json', 'rb+') as m: m.seek(-2, 2); m.write(b']')
    #         except: open(output+'.meta.json', 'a').write(']')
    #         with open(output+'.audio.json', 'rb+') as m: m.seek(-2, 2); m.write(b']')

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
