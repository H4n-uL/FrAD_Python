bits = 16
channels = 2
frame_size = 2048
sample_rate = 48000
desired_bitrate = 320000

uc = frame_size*bits*channels
tbr = int(desired_bitrate - (sample_rate / frame_size * 256))
cmp = int(tbr / sample_rate * frame_size)

print(f'Bits per frame uc:  {uc} bits')
print(f'Bits per frame cmp: {cmp} bits')
print(f'True Bitrate:       {tbr} bps')
print(f'Compression rate:   {cmp/uc*100:.3f} %')
