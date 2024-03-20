frame_size = 2048
sample_rate = 48000
desired_bitrate = 320000

print(f'Bits per frame: {int((desired_bitrate - (sample_rate / frame_size * 256)) / sample_rate * frame_size)} bits')
print(f'True Bitrate:   {int((desired_bitrate - (sample_rate / frame_size * 256)))} bps')
