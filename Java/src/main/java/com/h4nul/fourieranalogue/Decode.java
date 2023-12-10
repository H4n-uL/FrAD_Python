package com.h4nul.fourieranalogue;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
// import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.util.Arrays;
// import java.util.ArrayList;
// import java.util.List;

public class Decode {
    public void dec(String filePath, String out, int bits, String codec, String bitrate, Integer quality) throws Exception {
        if (bitrate == null) bitrate = "4096k";
        Decode decoder = new Decode();
    
        PCMRes PCM = decoder.internal(filePath, bits);
        byte[] restored = PCM.getPCMData();
        try (FileOutputStream output = new FileOutputStream("output.pcm")) {
            output.write(restored);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // int sampleRate = PCM.getSampleRate();
        // int channels = PCM.getChannels();
    
        // out = out != null ? out : "restored";
        // String[] split = out.split("\\.");
        // String container = split.length > 1 ? split[1].toLowerCase() : codec;
    
        // String format;
        // String sampleFormat;
        // switch (bits) {
        //     case 32: format = "s32le"; sampleFormat = "s32"; break;
        //     case 24: format = "s24le"; sampleFormat = "s24"; break;
        //     case 16: format = "s16le"; sampleFormat = "s16"; break;
        //     case 8:  format = sampleFormat = "u8"; break;
        //     default: throw new IllegalArgumentException("Illegal value " + bits + " for bits: only 8, 16, 24, and 32 bits are available for decoding.");
        // }
    
        // if (codec.equals("vorbis") || codec.equals("opus")) {
        //     codec = "lib" + codec;
        // }
    
        // List<String> command = new ArrayList<>();
        // command.add(FFpath.ffmpeg);
        // command.add("-y");
        // command.add("-f"); command.add(format);
        // command.add("-ar"); command.add(String.valueOf(sampleRate));
        // command.add("-ac"); command.add(String.valueOf(channels));
        // command.add("-i"); command.add("-");
        // command.add("-c:a"); command.add(codec);
    
        // if (Arrays.asList("pcm", "wav", "riff", "flac").contains(codec)) {
        //     command.add("-sample_fmt"); command.add(sampleFormat);
        // }
    
        // if (codec.equals("libvorbis")) {
        //     command.add("-q:a"); command.add(String.valueOf(quality));
        // }
    
        // if (Arrays.asList("aac", "m4a", "mp3", "libopus").contains(codec)) {
        //     if (codec.equals("libopus") && Integer.parseInt(bitrate.replace("k", "")) > 512) {
        //         bitrate = "512k";
        //     }
        //     command.add("-b:a"); command.add(bitrate);
        // }
    
        // command.add(out + "." + container);
    
        // ProcessBuilder processBuilder = new ProcessBuilder(command);
        // Process process = processBuilder.start();
        // OutputStream outputStream = process.getOutputStream();
        // outputStream.write(restored);
        // outputStream.close();
        // process.waitFor();
    }
    

    public PCMRes internal(String filePath, int bits) throws Exception {
        Fourier fourier = new Fourier();
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
            // replace with actual inverse FFT
            byte[] restored = fourier.Digital(data, fb, bits, channels);
            // return restored, sampleRate;
            return new PCMRes(restored, sampleRate, channels);
        }
    }
}
