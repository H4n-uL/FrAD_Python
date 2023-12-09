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
        byte[] blockLength = ByteBuffer.allocate(6).order(ByteOrder.LITTLE_ENDIAN).putInt(dataComb.length + 12).array();
        byte[] _block = Global.concat(COMMENT, blockLength, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(titleLength).array(), dataComb);
        return _block;
    }

    public static byte[] image(byte[] data) {
        byte[] blockLength = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(data.length + 10).array();
        byte[] _block = Global.concat(IMAGE, blockLength, data);
        return _block;
    }
}
