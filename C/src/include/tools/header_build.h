#ifndef HEADER_B_H
#define HEADER_B_H

#include <stdio.h>

int bits_to_b3(int bits);
unsigned char* builder(unsigned char* sample_rate_bytes, int channel, int bits, int is_ecc, unsigned char* md5);

#endif /* HEADER_B_H */
