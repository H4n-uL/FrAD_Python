from fva import encode
from fva import decode
from fva import header
from fva import player
from fva import repack

wav_name = 'mus.wav'
fva_name = 'fourierAnalogue.fva'
restored_name = 'restored.wav'

if __name__ == '__main__':
    encode.enc(wav_name, 32, out=fva_name, ecc_or_not=True, ecc_strength=0)
    # decode.dec(fva_name, out=restored_name)
    # print(header.parse(fva_name))
    # header.modify(fva_name)
    # repack.ecc(fva_name, ecc_type=0)
    # player.play(fva_name)
