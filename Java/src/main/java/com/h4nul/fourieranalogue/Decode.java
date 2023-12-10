package com.h4nul.fourieranalogue;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

import com.h4nul.fourieranalogue.tools.ECC;

public class Decode {
    public void dec(String filePath, String out, Integer bits, String codec, String quality) throws Exception {
        Decode decoder = new Decode();
        codec = (codec != null) ? codec : "flac";
        bits = (bits != null) ? bits : 32;

        PCMRes PCM = decoder.internal(filePath, bits);
        byte[] restored = PCM.getPCMData();
        int sampleRate = PCM.getSampleRate();
        int channels = PCM.getChannels();

        out = out != null ? out : "restored";
        String[] split = out.split("\\.");
        String container = split.length > 1 ? split[split.length - 1].toLowerCase() : codec;
        out = "";
        if (split.length == 1) {
            out = split[0] + ".";
        }
        else {
            for (int i = 0; i < split.length - 1; i++) {
                out = out + split[i] + ".";
            }
        }
        String format;
        String sampleFormat;
        switch (bits) {
            case 32: format = "s32le"; sampleFormat = "s32"; break;
            case 24: format = "s24le"; sampleFormat = "s24"; break;
            case 16: format = "s16le"; sampleFormat = "s16"; break;
            case 8:  format = sampleFormat = "u8"; break;
            default: throw new IllegalArgumentException("Illegal value " + bits + " for bits: only 8, 16, 24, and 32 bits are available for decoding.");
        }

        if (codec.equals("vorbis") || codec.equals("opus")) {
            codec = "lib" + codec;
        }

        List<String> command = new ArrayList<>();
        command.add(FFpath.ffmpeg);
        command.add("-y");
        command.add("-loglevel"); command.add("error");
        command.add("-f"); command.add(format);
        command.add("-ar"); command.add(String.valueOf(sampleRate));
        command.add("-ac"); command.add(String.valueOf(channels));
        command.add("-i"); command.add("-");
        command.add("-c:a"); command.add(codec);

        if (Arrays.asList("pcm", "wav", "riff", "flac").contains(codec)) {
            command.add("-sample_fmt"); command.add(sampleFormat);
        }

        if (codec.equals("libvorbis")) {
        if (quality == null) quality = "10";
            command.add("-q:a"); command.add(String.valueOf(quality));
        }

        if (Arrays.asList("aac", "m4a", "mp3", "libopus").contains(codec)) {
            if (quality == null) quality = "4096k";
            if (codec.equals("libopus") && Integer.parseInt(quality.replace("k", "")) > 512) {
                quality = "512k";
            }
            command.add("-b:a"); command.add(quality);
        }

        command.add(out + container);

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        Process process = processBuilder.start();

        Thread errorThread = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println(line);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        errorThread.start();

        // 이제 데이터를 쓸 수 있습니다.
        OutputStream outputStream = process.getOutputStream();
        outputStream.write(restored);
        process.waitFor();
        outputStream.close();
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
            int sampleRate = getInt24(ByteBuffer.wrap(Arrays.copyOfRange(header, 0x12, 0x16)).order(ByteOrder.LITTLE_ENDIAN).getInt());
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
                else {
                    byte[] restored = fourier.Digital(data, fb, bits, channels);
                    return new PCMRes(restored, sampleRate, channels);
                }
            } else {
                if (!Arrays.equals(checksumData, checksumHeader)) {
                    System.out.println(filePath + " has been corrupted, Please repack your file for the best music experience.");
                    System.out.println("Checksum: on header[" + Arrays.toString(checksumHeader) + "] vs on data[" + Arrays.toString(checksumData) + "]");
                    byte[] rawData = ECC.decode(data);
                    byte[] restored = fourier.Digital(rawData, fb, bits, channels);
                    return new PCMRes(restored, sampleRate, channels);
                }
                else {
                    byte[][] splitData = ECC.splitData(data, 148);
                    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
                    for (byte[] chunk : splitData) {
                        byte[] subChunk;
                        if (chunk.length >= 128) {subChunk = Arrays.copyOfRange(chunk, 0, 128);
                        } else {subChunk = chunk;}
                        outputStream.write(subChunk, 0, subChunk.length);
                    }
                    byte[] rawData = outputStream.toByteArray();
                    byte[] restored = fourier.Digital(rawData, fb, bits, channels);
                    return new PCMRes(restored, sampleRate, channels);
                }
            }
        }
    }

    private int getInt24(int i) {
        int value = 0;
        value |= (i & 0x00FFFFFF);
        return value;
    }
}
