package com.h4nul.fourieranalogue;

import com.h4nul.fourieranalogue.tools.HeaderB;

import ws.schild.jave.Encoder;
import ws.schild.jave.EncoderException;
import ws.schild.jave.InputFormatException;
import ws.schild.jave.MultimediaObject;
import ws.schild.jave.encode.AudioAttributes;
import ws.schild.jave.encode.EncodingAttributes;
import ws.schild.jave.info.AudioInfo;
import ws.schild.jave.info.MultimediaInfo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.Map;

public class Encode {
    public void enc(String filePath, int bits, String out, boolean applyEcc, Integer newSampleRate, Map<String, byte[]> meta, byte[] img) throws Exception {
        Fourier fourier = new Fourier();

        byte[] data = getPCM(filePath);
        int[] info = getInfo(filePath);
        int channels = info[0];
        int sampleRate = info[1];

        ByteBuffer sampleRateBytes = ByteBuffer.allocate(3);
        sampleRateBytes.order(ByteOrder.LITTLE_ENDIAN);
        int samp = (newSampleRate != null ? newSampleRate : sampleRate);
        sampleRateBytes.put((byte) (samp & 0xFF));
        sampleRateBytes.put((byte) ((samp >> 8) & 0xFF));
        sampleRateBytes.put((byte) ((samp >> 16) & 0xFF));

        // replace with actual Fourier transform
        data = fourier.Analogue(data, bits, channels, sampleRate, newSampleRate);
        // data = ecc.encode(data, applyEcc);

        MessageDigest md = MessageDigest.getInstance("MD5");
        md.update(data);  // replace with actual byte array
        byte[] checksum = md.digest();

        // replace with actual header builder
        byte[] h = HeaderB.uild(sampleRateBytes.array(), channels, bits, applyEcc, checksum, meta, img);

        if (out != null && (!out.endsWith(".fra") && !out.endsWith(".fva") && !out.endsWith(".sine"))) {
            out += ".fra";
        }

        try (FileOutputStream fos = new FileOutputStream(out != null ? out : "fourierAnalogue.fra")) {
            fos.write(h);
            fos.write(data);
        }
    }

    public byte[] getPCM(String path) throws IllegalArgumentException, InputFormatException, EncoderException, IOException {
        // Create a temporary file for output
        File target = File.createTempFile("temp", ".pcm");

        // Set audio attributes: PCM signed 32 bit, little endian
        AudioAttributes audio = new AudioAttributes();
        audio.setCodec("pcm_s32le");

        // Set encoding attributes
        EncodingAttributes attrs = new EncodingAttributes();
        attrs.setOutputFormat("s32le");
        attrs.setAudioAttributes(audio);

        // Encode source file to output stream
        Encoder encoder = new Encoder();
        encoder.encode(new MultimediaObject(new File(path)), target, attrs);

        // Read the temporary file into a byte array
        byte[] audioBytes = Files.readAllBytes(target.toPath());

        // Delete the temporary file
        target.delete();

        return audioBytes;
    }

    public int[] getInfo(String path) throws InputFormatException, EncoderException {
        // Get information about the source file
        MultimediaObject multimediaObject = new MultimediaObject(new File(path));

        // Get information about the source file
        MultimediaInfo info = multimediaObject.getInfo();
        AudioInfo audioInfo = info.getAudio();

        // Get sample rate and channels
        int sampleRate = audioInfo.getSamplingRate();
        int channels = audioInfo.getChannels();

        return new int[]{channels, sampleRate};
    }
}
