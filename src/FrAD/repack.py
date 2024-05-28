from .common import variables, methods
from .decoder import ASFH
from .encoder import encode
import os, shutil, struct, sys, time, zlib
from .tools.ecc import ecc
from .tools.headb import headb

class repack:
    @staticmethod
    def ecc(file_path, ecc_sizes: list | None = None, verbose: bool = False):
        with open(file_path, 'rb') as f:
            head = f.read(64)

            if methods.signature(head[0x0:0x4]) == 'container':
                head_len = struct.unpack('>Q', head[0x8:0x10])[0]
            else: head_len = 0
            f.seek(head_len)

            try:
                dlen = os.path.getsize(file_path) - head_len
                start_time = time.time()
                total_bytes = 0
                cli_width = 40
                fhead = None
                asfh = ASFH()
                with open(variables.temp, 'wb') as t:
                    if verbose: print('\n\n')
                    while True:
                        # Finding Audio Stream Frame Header
                        if fhead is None: fhead = f.read(4)
                        if fhead != b'\xff\xd0\xd2\x97':
                            hq = f.read(1)
                            if not hq: break
                            fhead = fhead[1:]+hq
                            continue

                        # Parsing ASFH
                        asfh.update(fhead+f.read(28))
                        # Reading Frame
                        frame = f.read(asfh.frmlen)

                        # Fixing errors and repacking
                        if asfh.ecc: frame = ecc.decode(frame, asfh.ecc_dsize, asfh.ecc_codesize)

                        if ecc_sizes is not None: ecc_dsize, ecc_codesize = ecc_sizes
                        elif asfh.ecc_dsize != 0 and asfh.ecc_codesize != 0: ecc_dsize, ecc_codesize = asfh.ecc_dsize, asfh.ecc_codesize
                        else: ecc_dsize, ecc_codesize = 96, 24

                        frame = ecc.encode(frame, ecc_dsize, ecc_codesize)

                        # EFloat Byte
                        efb = headb.encode_efb(asfh.profile, True, asfh.endian, asfh.float_bits)
                        encode.write_frame(t, frame, asfh.chnl, asfh.srate, efb, (ecc_dsize, ecc_codesize), asfh.fsize)

                        if verbose:
                            total_bytes += asfh.frmlen+32
                            elapsed_time = time.time() - start_time
                            bps = total_bytes / elapsed_time
                            percent = total_bytes * 100 / dlen
                            b = int(percent / 100 * cli_width)
                            eta = (elapsed_time / (percent / 100)) - elapsed_time if percent != 0 else 'infinity'
                            print('\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K', end='')
                            print(f'ECC Encode Speed: {(bps / 10**6):.3f} MB/s')
                            print(f'elapsed: {methods.tformat(elapsed_time)}, ETA {methods.tformat(eta)}')
                            print(f"[{'█'*b}{' '*(cli_width-b)}] {percent:.3f}% completed")

                        fhead = None
                    if verbose: print('\x1b[1A\x1b[2K', end='')

                f.seek(0)
                head = f.read(head_len)
            except KeyboardInterrupt:
                sys.exit(1)

            open(variables.temp2, 'wb').write(head+open(variables.temp, 'rb').read())
            shutil.move(variables.temp2, file_path)
