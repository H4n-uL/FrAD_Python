from fva import encoder
from fva import decoder
from fva import parser
from fva import headmod
from fva import player

encoder.encode('mus.wav', 32)
decoder.decode('fourierAnalogue.fva')
print(parser.parse('fourierAnalogue.fva'))
headmod.modify('fourierAnalogue.fva')
player.play('fourierAnalogue.fva')
