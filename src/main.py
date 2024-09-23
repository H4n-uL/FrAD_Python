from tools import cli
import encoder, decoder, repairer, header
import sys

'''               Fourier Analogue-in-Digital Master encoder/decoder
                             Original Author - HaמuL
'''

def main():
    ACTION, METAACTION, INPUT, PARAMS = cli.parse(sys.argv)

    if ACTION in cli.ENCODE_OPT:
        encoder.encode(INPUT, PARAMS)
    elif ACTION in cli.DECODE_OPT:
        decoder.decode(INPUT, PARAMS, False)
    elif ACTION in cli.PLAY_OPT:
        decoder.decode(INPUT, PARAMS, True)
    elif ACTION in cli.REPAIR_OPT:
        repairer.repair(INPUT, PARAMS)
    elif ACTION in cli.METADATA_OPT:
        header.modify(INPUT, METAACTION, PARAMS)

if __name__ == '__main__': main()