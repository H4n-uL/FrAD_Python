from .common import variables, methods
from .decoder import ASFH
from .encoder import encode
import os, shutil, struct, sys, time
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
                fhead, printed = None, False
                asfh = ASFH()
                with open(variables.temp, 'wb') as t:
                    while True:
                        # Finding Audio Stream Frame Header
                        if fhead is None: fhead = f.read(4)
                        if fhead != b'\xff\xd0\xd2\x97':
                            hq = f.read(1)
                            if not hq: break
                            fhead = fhead[1:]+hq
                            continue

                        # Parsing ASFH
                        asfh.update(f)
                        # Reading Frame
                        frame = f.read(asfh.frmbytes)

                        # Fixing errors and repacking
                        if asfh.ecc: frame = ecc.decode(frame, asfh.ecc_dsize, asfh.ecc_codesize)

                        if ecc_sizes is not None: ecc_dsize, ecc_codesize = ecc_sizes
                        elif asfh.ecc_dsize != 0 and asfh.ecc_codesize != 0: ecc_dsize, ecc_codesize = asfh.ecc_dsize, asfh.ecc_codesize
                        else: ecc_dsize, ecc_codesize = 96, 24

                        frame = ecc.encode(frame, ecc_dsize, ecc_codesize)

                        asfh.ecc = True
                        asfh.ecc_dsize, asfh.ecc_codesize = ecc_dsize, ecc_codesize
                        asfh.write_frame(t, frame)

                        if verbose:
                            total_bytes += asfh.frmbytes+asfh.headlen
                            elapsed_time = time.time() - start_time
                            bps = total_bytes / elapsed_time
                            percent = total_bytes * 100 / dlen
                            printed = methods.logging(3, 'ECC Encoding', printed, percent=percent, bps=bps, time=elapsed_time)
                        fhead = None

                f.seek(0)
                head = f.read(head_len)
            except KeyboardInterrupt:
                sys.exit(1)

            open(variables.temp2, 'wb').write(head+open(variables.temp, 'rb').read())
            shutil.move(variables.temp2, file_path)
