from .common import variables, methods
import hashlib, math, os, struct, sys, time, zlib
from .tools.ecc import ecc
from .tools.headb import headb

class repack:
    def ecc(file_path, verbose: bool = False):
        with open(file_path, 'rb') as f:
            head = f.read(64)

            methods.signature(head[0x0:0x4])

            header_length = struct.unpack('>Q', head[0x8:0x10])[0]
            efb = struct.unpack('<B', head[0x10:0x11])[0]
            is_ecc_on = True if (efb >> 4 & 0b1) == 0b1 else False # 0x10@0b100:    ECC Toggle(Enabled if 1)

            f.seek(header_length)

            try:
                dlen = os.path.getsize(file_path) - header_length
                start_time = time.time()
                total_bytes = 0
                cli_width = 40
                with open(variables.temp, 'wb') as t:
                    if verbose: print()
                    while True:
                        # Reading Frame Header
                        frame = f.read(32)
                        if not frame: break
                        blocklength = struct.unpack('>I', frame[0x4:0x8])[0]  # 0x04-4B: Audio Stream Frame length
                        efb = struct.unpack('>B', frame[0x8:0x9])[0]          # 0x08:    Cosine-Float Bit
                        is_ecc_on, float_bits = headb.decode_efb(efb)
                        channels = struct.unpack('>B', frame[0x9:0xa])[0] + 1 # 0x09:    Channels
                        ecc_dsize = struct.unpack('>B', frame[0xa:0xb])[0]    # 0x0a:    ECC Data block size
                        ecc_codesize = struct.unpack('>B', frame[0xb:0xc])[0] # 0x0b:    ECC Code size
                        srate_frame = struct.unpack('>I', frame[0xc:0x10])[0]        # 0x0c-4B: Sample rate
                        crc32 = frame[0x1c:0x20]                              # 0x1c-4B: ISO 3309 CRC32 of Audio Data

                        # Reading Block
                        block = f.read(blocklength)
                        total_bytes += blocklength

                        if is_ecc_on: block = ecc.decode(block, ecc_dsize, ecc_codesize)
                        else:
                            ecc_dsize = 128
                            ecc_codesize = 20
                        block = ecc.encode(block, ecc_dsize, ecc_codesize)
                        data = bytes(
                            #-- 0x00 ~ 0x0f --#
                                # Block Signature
                                b'\xff\xd0\xd2\x97' +

                                # Segment length(Processed)
                                struct.pack('>I', len(block)) +

                                headb.encode_efb(True, float_bits) + # EFB
                                struct.pack('>B', channels - 1) +    # Channels
                                struct.pack('>B', ecc_dsize) +       # ECC DSize
                                struct.pack('>B', ecc_codesize) +    # ECC code size

                                struct.pack('>I', srate_frame) +                     # Sample Rate

                            #-- 0x10 ~ 0x1f --#
                                b'\x00'*12 +

                                # ISO 3309 CRC32
                                struct.pack('>I', zlib.crc32(block)) +

                            #-- Data --#
                            block
                        )

                        # WRITE
                        t.write(data)

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

                f.seek(0)
                head = f.read(header_length)
            except KeyboardInterrupt:
                os.remove(variables.temp)
                sys.exit(1)

            with open(file_path, 'wb') as f, open(variables.temp, 'rb') as t:
                f.write(head)
                f.write(t.read())
            os.remove(variables.temp)
