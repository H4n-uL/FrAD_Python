#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "src/include/tools/global.h"
#include "src/include/tools/comment_block.h"

#define SIGNATURE "\x7e\x8b\xab\x89\xea\xc0\x9d\xa9\x68\x80"
#define RESERVED_LENGTH 217

int bits_to_b3(int bits) {
    switch (bits) {
        case 512: return 0b110;
        case 256: return 0b101;
        case 128: return 0b100;
        case 64: return 0b011;
        case 32: return 0b010;
        case 16: return 0b001;
        default: return 0b000;
    }
}

unsigned char* builder(unsigned char* sample_rate_bytes, int channel, int bits, int is_ecc, unsigned char* md5) {
    int b3 = bits_to_b3(bits);
    int cfb = ((channel - 1) << 3) | b3;
    unsigned char cfb_struct = (unsigned char)cfb;
    unsigned char ecc_bits = (unsigned char)((is_ecc ? 0b1 : 0b0) << 7);

    unsigned char* blocks = NULL;
    int blocks_length = 0;

    long length = strlen(SIGNATURE) + 8 + strlen((char*)sample_rate_bytes) + 1 + 1 + RESERVED_LENGTH + strlen((char*)md5) + blocks_length;
    unsigned char* header = (unsigned char*)malloc(length);
    memcpy(header, SIGNATURE, strlen(SIGNATURE));
    for (int i = 0; i < 8; i++) {
        header[strlen(SIGNATURE) + i] = (length >> (8 * i)) & 0xFF;
    }
    memcpy(header + strlen(SIGNATURE) + 8, sample_rate_bytes, strlen((char*)sample_rate_bytes));
    header[strlen(SIGNATURE) + 8 + strlen((char*)sample_rate_bytes)] = cfb_struct;
    header[strlen(SIGNATURE) + 8 + strlen((char*)sample_rate_bytes) + 1] = ecc_bits;
    for (int i = 0; i < RESERVED_LENGTH; i++) {
        header[strlen(SIGNATURE) + 8 + strlen((char*)sample_rate_bytes) + 1 + 1 + i] = '\x00';
    }
    memcpy(header + strlen(SIGNATURE) + 8 + strlen((char*)sample_rate_bytes) + 1 + 1 + RESERVED_LENGTH, md5, strlen((char*)md5));
    if (blocks != NULL) {
        memcpy(header + strlen(SIGNATURE) + 8 + strlen((char*)sample_rate_bytes) + 1 + 1 + RESERVED_LENGTH + strlen((char*)md5), blocks, blocks_length);
        free(blocks);
    }

    return header;
}
