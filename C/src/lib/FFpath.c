char* get_ffmpeg_path() {
#ifdef _WIN32
    return "./resources/codec/ffmpeg.Windows";
#elif __APPLE__
    return "./resources/codec/ffmpeg.macOS";
#elif __linux__
    #ifdef __x86_64__
        return "./resources/codec/ffmpeg.AMD64";
    #elif __aarch64__
        return "./resources/codec/ffmpeg.AArch64";
    #else
        printf("Unsupported architecture.");
        exit(EXIT_FAILURE);
    #endif
#else
    printf("Unsupported operating system.");
    exit(EXIT_FAILURE);
#endif
}