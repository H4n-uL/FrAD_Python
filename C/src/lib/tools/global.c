#include <stdio.h>
#include <string.h>
#include <stdlib.h>

unsigned char* concat(int count, unsigned char* arrays[], int lengths[]) {
    int length = 0;
    for (int i = 0; i < count; i++) {
        length += lengths[i];
    }
    unsigned char* result = (unsigned char*)malloc(sizeof(unsigned char) * length);
    int pos = 0;
    for (int i = 0; i < count; i++) {
        memcpy(result + pos, arrays[i], lengths[i]);
        pos += lengths[i];
    }
    return result;
}
