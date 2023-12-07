package com.h4nul.fourieranalogue;

import org.json.JSONArray;
import org.json.JSONObject;

import com.h4nul.fourieranalogue.tools.HeaderB;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.util.Map;

public class Encode {
    public static void enc(String filePath, int bits, String out, boolean applyEcc, Integer newSampleRate, Map<String, byte[]> meta, byte[] img) throws Exception {
        byte[] data = getPCM(filePath);
        int[] info = getInfo(filePath);
        int channels = info[0];
        int sampleRate = info[1];

        ByteBuffer sampleRateBytes = ByteBuffer.allocate(3);
        sampleRateBytes.order(ByteOrder.LITTLE_ENDIAN);
        sampleRateBytes.putShort((short) (newSampleRate != null ? newSampleRate : sampleRate));
        // replace with actual conversion to 32-bit integer PCM
        // data = ...

        // replace with actual Fourier transform
        data = Fourier.Analogue(data, bits, channels, sampleRate, newSampleRate);
        // data = ecc.encode(data, applyEcc);

        MessageDigest md = MessageDigest.getInstance("MD5");
        md.update(data);  // replace with actual byte array
        byte[] checksum = md.digest();

        // replace with actual header builder
        byte[] h = HeaderB.uild(sampleRateBytes.array(), channels, bits, applyEcc, checksum, meta, img);

        if (!out.endsWith(".fra") && !out.endsWith(".fva") && !out.endsWith(".sine")) {
            out += ".fra";
        }

        try (FileOutputStream fos = new FileOutputStream(out != null ? out : "fourierAnalogue.fra")) {
            fos.write(h);
            fos.write(data);
        }
    }

    public static int[] getInfo(String filePath) throws IOException, InterruptedException {
        ProcessBuilder builder = new ProcessBuilder(
                FFpath.ffprobe,
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                filePath
        );
        Process process = builder.start();
        process.waitFor();

        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder stringBuilder = new StringBuilder();
        String line = null;
        while ((line = reader.readLine()) != null) {
            stringBuilder.append(line);
        }
        String output = stringBuilder.toString();

        JSONObject jsonOutput = new JSONObject(output);
        JSONArray streams = jsonOutput.getJSONArray("streams");

        int channels = 0;
        int sampleRate = 0;

        for (int i = 0; i < streams.length(); i++) {
            JSONObject stream = streams.getJSONObject(i);
            if (stream.getString("codec_type").equals("audio")) {
                channels = stream.getInt("channels");
                sampleRate = stream.getInt("sample_rate");
                break;
            }
        }

        return new int[]{channels, sampleRate};
    }

    public static byte[] getPCM(String filePath) throws IOException, InterruptedException {
        ProcessBuilder builder = new ProcessBuilder(
                FFpath.ffmpeg,
                "-i", filePath,
                "-f", "s32le",
                "-acodec", "pcm_s32le",
                "-vn",
                "pipe:1"
        );
        Process process = builder.start();
        byte[] pcmData = process.getInputStream().readAllBytes();
        return pcmData;
    }
}
