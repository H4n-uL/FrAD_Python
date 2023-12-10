package com.h4nul.fourieranalogue;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PCMRes {
    private byte[] PCMData;
    private int sampleRate;
    private int channels;

    PCMRes(byte[] PCMData, int sampleRate, int channels) {
        this.PCMData = PCMData;
        this.sampleRate = sampleRate;
        this.channels = channels;
    }
}
