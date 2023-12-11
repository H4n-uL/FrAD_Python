#ifndef FOURIER_H
#define FOURIER_H

#include <stdio.h>

unsigned char* Analogue(unsigned char* data, int data_length, int bits, int channels, int osr, int* nsr);
unsigned char* Digital(unsigned char* data, int data_length, int fb, int bits, int channels);

#endif /* FOURIER_H */
