package com.h4nul.fourieranalogue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.apache.commons.math3.complex.Complex;
import org.jtransforms.fft.DoubleFFT_1D;

public class Fourier {
    public static byte[] Analogue(byte[] data, int bits, int channels, int osr, Integer nsr) {
        int numSamples = data.length / channels;

        // Convert the 1D PCM data to a 2D array.
        double[][] channelData = new double[channels][numSamples];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < channels; j++) {
                channelData[j][i] = (double) data[i * channels + j];
            }
        }

        // Perform FFT on each channel.
        DoubleFFT_1D fft = new DoubleFFT_1D(numSamples);
        for (int i = 0; i < channels; i++) {
            fft.realForward(channelData[i]);
        }

        // Get the amplitude and phase of each FFT result and convert them to byte[].
        ByteBuffer buffer = ByteBuffer.allocate(channels * numSamples * Float.BYTES * 2).order(ByteOrder.LITTLE_ENDIAN);  // 2 for real and imaginary parts
        for (int i = 0; i < channels; i++) {
            for (int j = 0; j < numSamples / 2; j++) {
                double real = channelData[i][j * 2];
                double imag = channelData[i][j * 2 + 1];
                buffer.putFloat((float) real);
                buffer.putFloat((float) imag);
            }
        }

        return buffer.array();
    }

    public static byte[] Digital(Complex[][] fftResults, int channels, int numSamples) {
        // Initialize the array for the inverse FFT results.
        double[][] channelData = new double[channels][numSamples];

        // Perform inverse FFT on each channel.
        DoubleFFT_1D fft = new DoubleFFT_1D(numSamples);
        for (int i = 0; i < channels; i++) {
            // Convert the Complex numbers back to a double array.
            for (int j = 0; j < numSamples / 2; j++) {
                channelData[i][j * 2] = fftResults[i][j].getReal();
                channelData[i][j * 2 + 1] = fftResults[i][j].getImaginary();
            }

            fft.realInverse(channelData[i], true);
        }

        // Convert the 2D array back to a 1D PCM data array.
        byte[] data = new byte[numSamples * channels];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < channels; j++) {
                data[i * channels + j] = (byte) channelData[j][i];
            }
        }

        return data;
    }
}
