from .common import variables, methods
import os, struct, sys, time, zlib
from .tools.ecc import ecc
from .tools.headb import headb

class repack:
    @staticmethod
    def ecc(file_path, ecc_sizes: list, verbose: bool = False):
        with open(file_path, 'rb') as f:
            head = f.read(64)

            methods.signature(head[0x0:0x4])

            header_length = struct.unpack('>Q', head[0x8:0x10])[0]

            f.seek(header_length)

            try:
                dlen = os.path.getsize(file_path) - header_length
                start_time = time.time()
                total_bytes = 0
                cli_width = 40
                with open(variables.temp, 'wb') as t:
                    if verbose: print('\n\n')
                    while True:
                        # Reading Frame Header
                        fhead = f.read(32)
                        if not fhead: break
                        framelength = struct.unpack('>I', fhead[0x4:0x8])[0]      # 0x04-4B: Audio Stream Frame length
                        efb = struct.unpack('>B', fhead[0x8:0x9])[0]              # 0x08:    Cosine-Float Bit
                        lossy, is_ecc_on, endian, float_bits = headb.decode_efb(efb)
                        channels = struct.unpack('>B', fhead[0x9:0xa])[0] + 1     # 0x09:    Channels
                        ed = struct.unpack('>B', fhead[0xa:0xb])[0]               # 0x0a:    ECC Data block size
                        ec = struct.unpack('>B', fhead[0xb:0xc])[0]               # 0x0b:    ECC Code size
                        srate_frame = struct.unpack('>I', fhead[0xc:0x10])[0]     # 0x0c-4B: Sample rate
                        samples_p_chnl = struct.unpack('>I', fhead[0x18:0x1c])[0] # 0x18-4B: Samples in a frame per channel
                        crc32 = fhead[0x1c:0x20]                                  # 0x1c-4B: ISO 3309 CRC32 of Audio Data
                        ssize_dict = {0b110: 128, 0b101: 64, 0b100: 48, 0b011: 32, 0b010: 24, 0b001: 16}

                        # Reading Frame
                        frame = f.read(framelength)

                        if is_ecc_on:
                            frame = ecc.decode(frame, ed, ec)
                            ecc_dsize = int(ecc_sizes[0])
                            ecc_codesize = int(ecc_sizes[1])
                        else:
                            ecc_dsize = 128
                            ecc_codesize = 20
                        frame = ecc.encode(frame, ecc_dsize, ecc_codesize)

                        efb = headb.encode_efb(lossy, True, endian, ssize_dict[float_bits])

                        data = bytes(
                            #-- 0x00 ~ 0x0f --#
                                # Frame Signature
                                b'\xff\xd0\xd2\x97' +

                                # Frame length(Processed)
                                struct.pack('>I', len(frame)) +

                                efb + # ECC-Float Byte
                                struct.pack('>B', channels - 1) + # Channels
                                struct.pack('>B', ecc_dsize) +    # ECC DSize
                                struct.pack('>B', ecc_codesize) + # ECC Code Size

                                # Sample Rate
                                struct.pack('>I', srate_frame) +

                            #-- 0x10 ~ 0x1f --#
                                b'\x00'*8 +

                                # Samples in a frame per channel
                                struct.pack('>I', samples_p_chnl) +

                                # ISO 3309 CRC32
                                struct.pack('>I', zlib.crc32(frame)) +

                            #-- Data --#
                            frame
                        )

                        # WRITE
                        t.write(data)

                        if verbose:
                            total_bytes += framelength+12
                            elapsed_time = time.time() - start_time
                            bps = total_bytes / elapsed_time
                            percent = total_bytes * 100 / dlen
                            b = int(percent / 100 * cli_width)
                            eta = (elapsed_time / (percent / 100)) - elapsed_time if percent != 0 else 'infinity'
                            print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                            print(f'ECC Encode Speed: {(bps / 10**6):.3f} MB/s')
                            print(f'elapsed: {methods.tformat(elapsed_time)}, ETA {methods.tformat(eta)}')
                            print(f"[{'█'*b}{' '*(cli_width-b)}] {percent:.3f}% completed")
                    if verbose: print('\x1b[1A\x1b[2K', end='')

                f.seek(0)
                head = f.read(header_length)
            except KeyboardInterrupt:
                sys.exit(1)

            with open(file_path, 'wb') as f, open(variables.temp, 'rb') as t:
                f.write(head)
                f.write(t.read())
