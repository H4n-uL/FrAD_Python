package com.h4nul.fourieranalogue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.jtransforms.fft.DoubleFFT_1D;

public class Fourier {
    public byte[] Analogue(byte[] data, int bits, int channels, int osr, Integer nsr) {
        int numSamples = data.length / channels / 4;
        double[][] channelData = new double[channels][2 * numSamples];
        ByteBuffer buffer = ByteBuffer.wrap(data);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < channels; j++) {
                channelData[j][i] = (double) buffer.getInt();
            }
        }

        if (nsr != null && nsr != osr) {
            System.out.println("Resampling is not supported on java yet.");
            // TODO: Add resampling code here.
        }

        DoubleFFT_1D fft = new DoubleFFT_1D(numSamples);
        double[][] fft_data = new double[channels][2 * numSamples];
        for (int i = 0; i < channels; i++) {
            fft.realForwardFull(channelData[i]);
            fft_data[i] = channelData[i];
        }

        byte[] output = new byte[numSamples * channels * (bits / 4)];
        int index = 0;

        for (int j = 0; j < numSamples; j++) {
            for (int i = 0; i < channels; i++) {
                double ampli = Math.hypot(fft_data[i][2 * j], fft_data[i][2 * j + 1]);
                double phase = Math.atan2(fft_data[i][2 * j + 1], fft_data[i][2 * j]);
                switch (bits) {
                    case 64:
                        byte[] ampliBytes = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putDouble(ampli).array();
                        byte[] phaseBytes = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putDouble(phase).array();
                        System.arraycopy(ampliBytes, 0, output, index, 8);
                        System.arraycopy(phaseBytes, 0, output, index + 8, 8);
                        index += 16;
                        break;
                    case 32:
                        byte[] ampliBytesFloat = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat((float) ampli).array();
                        byte[] phaseBytesFloat = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat((float) phase).array();
                        System.arraycopy(ampliBytesFloat, 0, output, index, 4);
                        System.arraycopy(phaseBytesFloat, 0, output, index + 4, 4);
                        index += 8;
                        break;
                    // case 16:
                    //     byteBuffer.putShort(floatToBFloat16(ampli));
                    //     byteBuffer.putShort(floatToBFloat16(phase));
                    //     break;
                    default:
                        throw new IllegalArgumentException("Illegal bits value.");
                }
            }
        }
        return output;
    }

    public byte[] Digital(byte[] data, int fb, int bits, int channels) {
        int numSamples = data.length / (channels * 2 * ((int) Math.pow(2, fb + 3) / 8));
        ByteBuffer buffer = ByteBuffer.wrap(data);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        double[][] freqData = new double[channels][numSamples * 2];

        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < channels; j++) {
                double ampli = buffer.getDouble();
                double phase = buffer.getDouble();
                freqData[j][2 * i] = ampli * Math.cos(phase);
                freqData[j][2 * i + 1] = ampli * Math.sin(phase);
            }
        }

        double[][] waveData = new double[channels][numSamples];
        DoubleFFT_1D ifft = new DoubleFFT_1D(numSamples);

        for (int i = 0; i < channels; i++) {
            ifft.complexInverse(freqData[i], true);
            for (int j = 0; j < numSamples; j++) {
                waveData[i][j] = freqData[i][2 * j];  // Ignore the imaginary part
            }
        }

        byte[] output = new byte[numSamples * channels * (bits / 8)];
        int index = 0;

        for (int j = 0; j < numSamples; j++) {
            for (int i = 0; i < channels; i++) {
                switch (bits) {
                    case 32:
                        byte[] waveBytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt((int) waveData[i][j]).array();
                        System.arraycopy(waveBytes, 0, output, index, 4);
                        index += 4;
                        break;
                    case 16:
                        byte[] waveBytesShort = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort((short) (waveData[i][j] / (1 << 16))).array();
                        System.arraycopy(waveBytesShort, 0, output, index, 2);
                        index += 2;
                        break;
                    case 8:
                        output[index++] = (byte) ((waveData[i][j] / (1 << 24)) + (1 << 7));
                        break;
                    default:
                        throw new IllegalArgumentException("Illegal bits value.");
                }
            }
        }
        return output;
    }
}    
