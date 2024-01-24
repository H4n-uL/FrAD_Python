from .common import variables, ecc_v, methods
import hashlib, math, os, struct, sys, time, zlib
from .tools.ecc import ecc

class repack:
    def ecc(file_path, verbose: bool = False):
        with open(file_path, 'rb') as f:
            try:
                head = f.read(256)

                methods.signature(head[0x0:0x3])

                header_length = struct.unpack('>Q', head[0x8:0x10])[0]
                efb = struct.unpack('<B', head[0x10:0x11])[0]
                is_secure = True if (efb >> 5 & 0b1) == 0b1 else False # 0x10@0b101:    Secure frame Toggle(Enabled if 1)
                is_ecc_on = True if (efb >> 4 & 0b1) == 0b1 else False # 0x10@0b100:    ECC Toggle(Enabled if 1)
                fsize = struct.unpack('<B', head[0x11:0x12])[0] >> 5   # 0x11@0b111-4b: Frame size

                f.seek(header_length)
                variables.nperseg = int(math.pow(2, fsize + 7))
                nperseg = (variables.nperseg if not is_ecc_on else variables.nperseg // ecc_v.data_size * ecc_v.block_size) * 16384
            except KeyboardInterrupt:
                sys.exit(1)
            try:
                dlen = os.path.getsize(file_path) - header_length
                start_time = time.time()
                total_bytes = 0
                cli_width = 40
                with open(variables.temp, 'wb') as t:
                    if verbose: print()
                    while True:
                        if is_secure:
                            frame = f.read(10)
                            if not frame: break
                            blocklength = struct.unpack('>I', frame[0x2:0x6])[0]
                            block = f.read(blocklength)
                            if zlib.crc32(block) != struct.unpack('>I', frame[0x6:0x10])[0]:
                                block = b'\x00'*blocklength
                            # block = zlib.decompress(block)
                            total_bytes += blocklength
                        else:
                            block = f.read(nperseg)
                            if not block: break
                            total_bytes += len(block)

                        if is_ecc_on: block = ecc.decode(block)
                        block = ecc.encode(block)

                        if is_secure:
                            t.write(b'\xff\x0f' + struct.pack('>I', len(block)) + struct.pack('>I', zlib.crc32(block)) + block)
                        else:
                            t.write(block)

                        if verbose:
                            total_bytes += len(block)
                            elapsed_time = time.time() - start_time
                            bps = total_bytes / elapsed_time
                            percent = (total_bytes * 100 // (1 if is_ecc_on else (32/37))) / dlen
                            b = int(percent / 100 * cli_width)
                            print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                            print(f'ECC Encode Speed: {(bps / 10**6):.3f} MB/s')
                            print(f"[{'█'*b}{' '*(cli_width-b)}] {percent:.3f}% completed")
                    if verbose: print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')

                with open(variables.temp, 'rb') as t:
                    md5 = hashlib.md5()
                    while True:
                        d = t.read(variables.hash_block_size)
                        if not d: break
                        md5.update(d)
                    checksum = md5.digest()

                f.seek(0)
                head = bytearray(f.read(header_length))
                head[0xf0:0x100] = checksum
                head[0x10:0x11] = struct.pack('<B', 0b1 << 4 | efb)
                head = bytes(head)
            except KeyboardInterrupt:
                os.remove(variables.temp)
                sys.exit(1)

            with open(file_path, 'wb') as f:
                f.write(head)
                with open(variables.temp, 'rb') as t:
                    while True:
                        a = t.read(1048576)
                        if not a: break
                        f.write(a)
            os.remove(variables.temp)
