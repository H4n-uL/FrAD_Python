package com.h4nul.fourieranalogue.tools;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

public class CommentBlock {
    static final byte[] IMAGE = {(byte)0xf5, (byte)0x55};
    static final byte[] COMMENT = {(byte)0xfa, (byte)0xaa};

    public static byte[] comment(String title, byte[] data) {
        int titleLength = title.length();
        byte[] dataComb = Global.concat(title.getBytes(StandardCharsets.UTF_8), data);

        ByteBuffer blengthBuffer = ByteBuffer.allocate(6);
        blengthBuffer.order(ByteOrder.LITTLE_ENDIAN);
        long blength = dataComb.length + 12;
        blengthBuffer.put((byte) (blength & 0xFF));
        blengthBuffer.put((byte) ((blength >> 8) & 0xFF));
        blengthBuffer.put((byte) ((blength >> 16) & 0xFF));
        blengthBuffer.put((byte) ((blength >> 24) & 0xFF));
        blengthBuffer.put((byte) ((blength >> 32) & 0xFF));
        blengthBuffer.put((byte) ((blength >> 40) & 0xFF));

        byte[] blockLength = blengthBuffer.array();
        byte[] _block = Global.concat(COMMENT, blockLength, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(titleLength).array(), dataComb);
        return _block;
    }

    public static byte[] image(byte[] data) {
        byte[] blockLength = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(data.length + 10).array();
        byte[] _block = Global.concat(IMAGE, blockLength, data);
        return _block;
    }
}
