package com.h4nul.fourieranalogue;

import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.util.Arrays;

import org.apache.commons.math3.complex.Complex;

public class Decode {
    public static PCMRes internal(String filePath, int bits) throws Exception {
        try (FileInputStream fis = new FileInputStream(filePath)) {
            byte[] header = fis.readNBytes(256);

            byte[] signature = Arrays.copyOfRange(header, 0x0, 0xa);
            if (!Arrays.equals(signature, new byte[]{0x7e, (byte) 0x8b, (byte) 0xab, (byte) 0x89, (byte) 0xea, (byte) 0xc0, (byte) 0x9d, (byte) 0xa9, 0x68, (byte) 0x80})) {
                throw new Exception("This is not Fourier Analogue file.");
            }

            long headerLength = ByteBuffer.wrap(Arrays.copyOfRange(header, 0xa, 0x12)).order(ByteOrder.LITTLE_ENDIAN).getLong();
            int sampleRate = ByteBuffer.wrap(Arrays.copyOfRange(header, 0x12, 0x15)).order(ByteOrder.LITTLE_ENDIAN).getShort();
            byte cfb = header[0x15];
            int channels = (cfb >> 3) + 1;
            int fb = cfb & 0b111;
            boolean isEccOn = (header[0x16] & 0x80) != 0;
            byte[] checksumHeader = Arrays.copyOfRange(header, 0xf0, 0x100);

            fis.skip(headerLength - 256);

            byte[] data = fis.readAllBytes();
            MessageDigest md = MessageDigest.getInstance("MD5");
            md.update(data);
            byte[] checksumData = md.digest();
            if (!isEccOn) {
                if (!Arrays.equals(checksumData, checksumHeader)) {
                    System.out.println("Checksum: on header[" + Arrays.toString(checksumHeader) + "] vs on data[" + Arrays.toString(checksumData) + "]");
                    throw new Exception("File has corrupted but it has no ECC option. Decoder halted.");
                }
            } else {
                if (!Arrays.equals(checksumData, checksumHeader)) {
                    System.out.println(filePath + " has been corrupted, Please repack your file for the best music experience.");
                    System.out.println("Checksum: on header[" + Arrays.toString(checksumHeader) + "] vs on data[" + Arrays.toString(checksumData) + "]");
                    // replace with actual ECC decoding
                    // data = ecc.decode(data);
                }
                // replace with actual ECC decoding
                // data = ecc.decode(data);
            }

            Complex[][] complexData;
            if (fb == 0b011) complexData = fromDoubleBytesToComplex(data, Double.BYTES);
            else if (fb == 0b010) complexData = fromFloatBytesToComplex(data, channels);
            // else if (fb == 0b001) data = fromFloatArray(data, Float.BYTES / 2);
            else throw new Exception("Illegal bits value.");

            // replace with actual inverse FFT
            byte[] restored = Fourier.Digital(complexData, bits, channels);
            // return restored, sampleRate;
            return new PCMRes(restored, sampleRate);
        }
    }

    private static Complex[][] fromDoubleBytesToComplex(byte[] data, int channels) {
        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int numSamples = data.length / Double.BYTES / 2 / channels;

        Complex[][] complexData = new Complex[channels][numSamples];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < channels; j++) {
                double real = buffer.getDouble();
                double imag = buffer.getDouble();
                complexData[j][i] = new Complex(real, imag);
            }
        }

        return complexData;
    }

    private static Complex[][] fromFloatBytesToComplex(byte[] data, int channels) {
        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int numSamples = data.length / Float.BYTES / 2 / channels;

        Complex[][] complexData = new Complex[channels][numSamples];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < channels; j++) {
                float real = buffer.getFloat();
                float imag = buffer.getFloat();
                complexData[j][i] = new Complex(real, imag);
            }
        }

        return complexData;
    }
}

