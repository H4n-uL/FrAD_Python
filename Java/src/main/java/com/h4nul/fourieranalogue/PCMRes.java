package com.h4nul.fourieranalogue;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PCMRes {
    private byte[] PCMData;
    private int sampleRate;

    PCMRes(byte[] PCMData, int sampleRate) {
        this.PCMData = PCMData;
        this.sampleRate = sampleRate;
    }
}
