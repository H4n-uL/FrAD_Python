from fra import encode
from fra import decode
from fra import header
from fra import player
from fra import repack

wav_name = 'audio.flac'
fra_name = 'fourierAnalogue.fra'
restored_name = 'restored.flac'

meta = [
    ['ARTIST', 'HaמuL'],
    ['TITLE', 'Fourier Analogue']
]

if __name__ == '__main__':
    encode.enc(wav_name, 64, out=fra_name, meta=meta)
    decode.dec(fra_name, out=restored_name, codec='flac')
    head, _ = header.parse(fra_name)
    print(head)
    header.modify(fra_name)
    repack.ecc(fra_name)
    player.play(fra_name)
