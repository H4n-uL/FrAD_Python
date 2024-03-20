frame_size = 2048
sample_rate = 48000
desired_bitrate = 320000

print(f'Available bits per frame: {int((desired_bitrate - (sample_rate / frame_size * 256)) / sample_rate * frame_size)} bits')
# 13397 bits for 2048 spf, 48000 Hz, 320 kbps
