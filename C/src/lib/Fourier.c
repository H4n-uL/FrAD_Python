#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

unsigned char* Analogue(unsigned char* data, int* data_length, int bits, int channels, int osr, int* nsr) {
    int num_samples = *data_length / channels / 4;
    double** channel_data = (double**)malloc(channels * sizeof(double*));
    for (int i = 0; i < channels; i++) {
        channel_data[i] = (double*)malloc(2 * num_samples * sizeof(double));
    }

    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < channels; j++) {
            int value = 0;
            for (int k = 0; k < 4; k++) {
                value |= data[i * channels * 4 + j * 4 + k] << (8 * k);
            }
            channel_data[j][i] = (double)value;
        }
    }

    if (nsr != NULL && *nsr != osr) {
        printf("Resampling is not supported on C yet.\n");
        // TODO: Add resampling code here.
    }

    fftw_complex** fft_data = (fftw_complex**)malloc(channels * sizeof(fftw_complex*));
    for (int i = 0; i < channels; i++) {
        fft_data[i] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * num_samples);
        fftw_plan plan = fftw_plan_dft_r2c_1d(num_samples, channel_data[i], fft_data[i], FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    unsigned char* output = (unsigned char*)malloc(num_samples * channels * (bits / 4));
    int index = 0;

    for (int j = 0; j < num_samples; j++) {
        for (int i = 0; i < channels; i++) {
            double ampli = sqrt(pow(creal(fft_data[i][j]), 2) + pow(cimag(fft_data[i][j]), 2));
            double phase = atan2(cimag(fft_data[i][j]), creal(fft_data[i][j]));
            switch (bits) {
                case 64:
                    for (int k = 0; k < 8; k++) {
                        output[index + k] = ((unsigned long long)ampli >> (8 * k)) & 0xFF;
                        output[index + 8 + k] = ((unsigned long long)phase >> (8 * k)) & 0xFF;
                    }
                    index += 16;
                    break;
                case 32:
                    for (int k = 0; k < 4; k++) {
                        output[index + k] = ((unsigned int)ampli >> (8 * k)) & 0xFF;
                        output[index + 4 + k] = ((unsigned int)phase >> (8 * k)) & 0xFF;
                    }
                    index += 8;
                    break;
                // case 16:
                //     byteBuffer.putShort(floatToBFloat16(ampli));
                //     byteBuffer.putShort(floatToBFloat16(phase));
                //     break;
                default:
                    printf("Illegal bits value.\n");
                    exit(EXIT_FAILURE);
            }
        }
    }

    for (int i = 0; i < channels; i++) {
        fftw_free(fft_data[i]);
        free(channel_data[i]);
    }
    free(fft_data);
    free(channel_data);
    *data_length = num_samples*(bits/8);
    return output;
}

unsigned char* Digital(unsigned char* data, int data_length, int fb, int bits, int channels) {
    int bi = (int)pow(2, fb + 3);
    int num_samples = data_length / (channels * 2 * (bi / 8));

    double** freq_data = (double**)malloc(channels * sizeof(double*));
    for (int i = 0; i < channels; i++) {
        freq_data[i] = (double*)malloc(num_samples * 2 * sizeof(double));
    }

    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < channels; j++) {
            if (bi == 64) {
                double ampli = 0;
                double phase = 0;
                for (int k = 0; k < 8; k++) {
                    ampli += data[i * channels * 16 + j * 16 + k] << (8 * k);
                    phase += data[i * channels * 16 + j * 16 + 8 + k] << (8 * k);
                }
                freq_data[j][2 * i] = ampli * cos(phase);
                freq_data[j][2 * i + 1] = ampli * sin(phase);
            }
            if (bi == 32) {
                float ampli = 0;
                float phase = 0;
                for (int k = 0; k < 4; k++) {
                    ampli += data[i * channels * 8 + j * 8 + k] << (8 * k);
                    phase += data[i * channels * 8 + j * 8 + 4 + k] << (8 * k);
                }
                freq_data[j][2 * i] = ampli * cos(phase);
                freq_data[j][2 * i + 1] = ampli * sin(phase);
            }
        }
    }

    double** wave_data = (double**)malloc(channels * sizeof(double*));
    for (int i = 0; i < channels; i++) {
        wave_data[i] = (double*)malloc(num_samples * sizeof(double));
        fftw_plan plan = fftw_plan_dft_c2r_1d(num_samples, freq_data[i], wave_data[i], FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    unsigned char* output = (unsigned char*)malloc(num_samples * channels * (bits / 8));
    int index = 0;

    for (int j = 0; j < num_samples; j++) {
        for (int i = 0; i < channels; i++) {
            switch (bits) {
                case 32:
                    for (int k = 0; k < 4; k++) {
                        output[index + k] = ((unsigned int)wave_data[i][j] >> (8 * k)) & 0xFF;
                    }
                    index += 4;
                    break;
                case 16:
                    for (int k = 0; k < 2; k++) {
                        output[index + k] = ((unsigned short)(wave_data[i][j] / (1 << 16)) >> (8 * k)) & 0xFF;
                    }
                    index += 2;
                    break;
                case 8:
                    output[index++] = (unsigned char)((wave_data[i][j] / (1 << 24)) + (1 << 7));
                    break;
                default:
                    printf("Illegal bits value.");
                    exit(EXIT_FAILURE);
            }
        }
    }

    for (int i = 0; i < channels; i++) {
        fftw_free(freq_data[i]);
        free(wave_data[i]);
    }
    free(freq_data);
    free(wave_data);

    return output;
}
