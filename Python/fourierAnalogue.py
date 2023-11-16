from fva import encoder
from fva import decoder
from fva import parser
from fva import headmod
from fva import player

wav_name = 'mus.wav'
fva_name = 'fourierAnalogue.fva'
restored_name = 'restored.wav'

if __name__ == '__main__':
    encoder.encode(wav_name, 32, out=fva_name)
    decoder.decode(fva_name, out=restored_name)
    print(parser.parse(fva_name))
    headmod.modify(fva_name)
    player.play(fva_name)
