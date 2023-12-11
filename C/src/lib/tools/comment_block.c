#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "src/include/tools/global.h"

#define IMAGE "\xf5\x55"
#define COMMENT "\xfa\xaa"

unsigned char* comment(char* title, unsigned char* data, int data_length) {
    int title_length = strlen(title);
    unsigned char* title_bytes = (unsigned char*)title;
    unsigned char* data_comb = concat(2, (unsigned char*[]){title_bytes, data}, (int[]){title_length, data_length});

    long blength = title_length + data_length + 12;
    unsigned char block_length[6];
    for (int i = 0; i < 6; i++) {
        block_length[i] = (blength >> (8 * i)) & 0xFF;
    }

    unsigned char title_length_bytes[4];
    for (int i = 0; i < 4; i++) {
        title_length_bytes[i] = (title_length >> (8 * i)) & 0xFF;
    }
    
    unsigned char* _block = concat(4, (unsigned char*[]){(unsigned char*)COMMENT, block_length, title_length_bytes, data_comb},
                                   (int[]){2, 6, 4, title_length + data_length});
    free(data_comb);
    return _block;
}

unsigned char* image(unsigned char* data, int data_length) {
    unsigned char block_length[8];
    for (int i = 0; i < 8; i++) {
        block_length[i] = ((data_length + 10) >> (8 * i)) & 0xFF;
    }

    unsigned char* _block = concat(3, (unsigned char*[]){(unsigned char*)IMAGE, block_length, data},
                                   (int[]){2, 8, data_length});
    return _block;
}
