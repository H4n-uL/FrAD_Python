bits = 16
channels = 2
frame_size = 2048
overlap_ratio_th = 16
sample_rate = 48000
desired_bitrate = 320000
fixedheader = 32
lossyheader = 16

uc = frame_size*bits*channels
tbr = int(desired_bitrate + (sample_rate / frame_size * (fixedheader + lossyheader)*8))
cmp = int(desired_bitrate / sample_rate * frame_size // 8 // channels * 8 * channels)

print(f'Bits per frame uncmp:  {uc} bits')
print(f'Bits per frame cmp:    {cmp} bits')
print(f'True Bitrate:          {tbr} bps')
print(f'Compression rate:      {100-cmp/uc*100:.3f} %')
print(f'True/Desired br ratio: {tbr/desired_bitrate*100:.3f} %')

# 헤더 오버헤드 알빠노 ㅋㅋ
