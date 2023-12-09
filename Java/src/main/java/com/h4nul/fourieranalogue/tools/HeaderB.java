package com.h4nul.fourieranalogue.tools;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

public class HeaderB {
    static final byte[] SIGNATURE = {(byte) 0x7e, (byte) 0x8b, (byte) 0xab, (byte) 0x89, (byte) 0xea, (byte) 0xc0, (byte) 0x9d, (byte) 0xa9, (byte) 0x68, (byte) 0x80};
    static final byte[] RESERVED = new byte[217];

    static final Map<Integer, Integer> BITS_TO_B3 = new HashMap<Integer, Integer>() {{
        put(512, 0b110);
        put(256, 0b101);
        put(128, 0b100);
        put(64, 0b011);
        put(32, 0b010);
        put(16, 0b001);
    }};

    public static byte[] uild(
            byte[] sampleRateBytes, int channel, int bits, boolean isEcc, byte[] md5, Map<String, byte[]> meta, byte[] img) {
        int b3 = BITS_TO_B3.getOrDefault(bits, 0b000);

        int cfb = ((channel - 1) << 3) | b3;

        byte[] cfbStruct = ByteBuffer.allocate(1).order(ByteOrder.LITTLE_ENDIAN).put((byte) cfb).array();
        byte[] eccBits = ByteBuffer.allocate(1).order(ByteOrder.LITTLE_ENDIAN).put((byte) ((isEcc ? 0b1 : 0b0) << 7 | 0b0000000)).array();

        byte[] blocks = new byte[0];

        if (meta != null) {
            for (Map.Entry<String, byte[]> entry : meta.entrySet()) {
                blocks = Global.concat(blocks, CommentBlock.comment(entry.getKey(), entry.getValue()));
            }
        }
        if (img != null) {
            blocks = Global.concat(blocks, CommentBlock.image(img));
        }

        byte[] length = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(SIGNATURE.length + 8 + sampleRateBytes.length + cfbStruct.length + eccBits.length + RESERVED.length + md5.length + blocks.length).array();

        byte[] header = Global.concat(SIGNATURE, length, sampleRateBytes, cfbStruct, eccBits, RESERVED, md5, blocks);

        return header;
    }
}
