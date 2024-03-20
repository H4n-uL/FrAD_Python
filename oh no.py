bits = 16
channels = 2
frame_size = 2048
sample_rate = 48000
desired_bitrate = 8000

uc = frame_size*bits*channels
tbr = int(desired_bitrate - (sample_rate / frame_size * 256))
cmp = int(tbr / sample_rate * frame_size // 8 * 8)

print(f'Bits per frame uncmp:  {uc} bits')
print(f'Bits per frame cmp:    {cmp} bits')
print(f'True Bitrate:          {tbr} bps')
print(f'Compression rate:      {cmp/uc*100:.3f} %')
print(f'True/Desired br ratio: {tbr/desired_bitrate*100:.3f} %')
