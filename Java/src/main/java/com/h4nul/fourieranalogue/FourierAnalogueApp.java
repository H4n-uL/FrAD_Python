package com.h4nul.fourieranalogue;

public class FourierAnalogueApp {
    public static void main(String[] args) throws Exception {
        // Encode encode = new Encode();
        Decode decode = new Decode();

        String fra = "fourierAnalogue.java.fra";
        String flac = "restored.flac";
        // encode.enc("/Users/H4nUL/Desktop/cks.flac", 64, fra, false, null, null, null);
        decode.dec(fra, flac, 32, null, null);
    }
}
