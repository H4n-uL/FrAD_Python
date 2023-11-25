from fva import encode
from fva import decode
from fva import header
from fva import player
from fva import repack

wav_name = 'audio.flac'
fra_name = 'fourierAnalogue.fra'
restored_name = 'restored.flac'

if __name__ == '__main__':
    encode.enc(wav_name, 32, out=fra_name)
    decode.dec(fra_name, out=restored_name)
    print(header.parse(fra_name))
    header.modify(fra_name)
    repack.ecc(fra_name)
    player.play(fra_name)
