package com.h4nul.fourieranalogue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.cli.*;

public class FourierAnalogueApp {
    public static void main(String[] args) throws Exception {
        Options options = new Options();

        Option input = new Option("i", "input", true, "input file path");
        input.setRequired(true);
        options.addOption(input);

        Option output = new Option("o", "output", true, "output file path");
        output.setRequired(false);
        options.addOption(output);

        Option bits = new Option("b", "bits", true, "output file bit depth");
        bits.setRequired(false);
        options.addOption(bits);

        Option applyEcc = new Option("ecc", "applyecc", false, "apply ecc");
        applyEcc.setRequired(false);
        options.addOption(applyEcc);

        Option img = new Option("img", "image", true, "image file path");
        img.setRequired(false);
        options.addOption(img);

        Option codec = new Option("c", "codec", true, "codec type");
        codec.setRequired(false);
        options.addOption(codec);

        Option meta = Option.builder("m")
                .longOpt("meta")
                .desc("metadata in \"key\" \"value\" format")
                .hasArgs()
                .build();
        options.addOption(meta);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("FourierAnalogueApp", options);

            System.exit(1);
            return;
        }

        String inputFilePath = cmd.getOptionValue("input");
        String outputFilePath = cmd.getOptionValue("output");
        String b = cmd.getOptionValue("bits");
        Integer bitsValue = null;
        if (b != null) {
            bitsValue = Integer.parseInt(cmd.getOptionValue("bits"));
        }
        boolean isecc = cmd.hasOption("ecc");
        String imageFilePath = cmd.getOptionValue("img");
        String codecType = cmd.getOptionValue("codec");
        String[] metaValues = cmd.getOptionValues("meta");

        Map<String, byte[]> metaMap = new HashMap<>();
        if (metaValues != null) {
            for (int i = 0; i < metaValues.length; i += 2) {
                String key = metaValues[i];
                byte[] value = metaValues[i+1].getBytes(StandardCharsets.UTF_8);
                metaMap.put(key, value);
            }
        }

        byte[] imageBytes = null;
        if (imageFilePath != null) {
            Path imagePath = Paths.get(imageFilePath);
            imageBytes = Files.readAllBytes(imagePath);
        }

        if (args[0].equals("encode")) {
            Encode encode = new Encode();
            if (bitsValue == null) throw new IllegalArgumentException();
            encode.enc(inputFilePath, bitsValue, outputFilePath, isecc, null, metaMap, imageBytes);
        }
        else if (args[0].equals("decode")) {
            Decode decode = new Decode();
            decode.dec(inputFilePath, outputFilePath, bitsValue, codecType, null);
        }
    }
}
