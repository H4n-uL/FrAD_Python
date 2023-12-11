#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <openssl/md5.h>
#include "src/include/Fourier.h"
#include "src/include/tools/header_build.h"

void enc(const char* file_path, int bits, const char* out_path, int apply_ecc, int new_sample_rate) {
    int data_size;
    uint8_t* data = get_pcm(file_path, &data_size);
    AudioInfo info = get_info(file_path);

    // Fourier 변환
    int nsr;
    unsigned char* transformed_data = Analogue(data, &data_size, bits, info.channels, info.sample_rate, &nsr);
    data_size = sizeof(transformed_data);

    // ECC 적용 및 체크섬 생성
    // ... 이 부분은 ECC.encode와 MD5 체크섬 생성에 해당하는 C 코드가 필요합니다 ...
    unsigned char checksum[MD5_DIGEST_LENGTH];
    MD5(transformed_data, data_size, checksum);

    // HeaderB.uild에 해당하는 헤더 생성
    unsigned char sample_rate_bytes[3];
    int sample_rate = new_sample_rate ? new_sample_rate : info.sample_rate;
    sample_rate_bytes[0] = sample_rate & 0xFF;
    sample_rate_bytes[1] = (sample_rate >> 8) & 0xFF;
    sample_rate_bytes[2] = (sample_rate >> 16) & 0xFF;

    unsigned char* h = builder(sample_rate_bytes, info.channels, bits, apply_ecc, checksum);
    int h_size = sizeof(h);
    // ... 이 부분은 HeaderB.uild에 해당하는 C 코드가 필요합니다 ...

    // 파일 이름 설정
    char* file_name;
    if (strstr(out_path, ".fra") || strstr(out_path, ".fva") || strstr(out_path, ".sine")) {
        file_name = (char*)out_path;
    } else {
        file_name = strcat((char*)out_path, ".fra");
    }

    // 파일 저장
    FILE* file = fopen(file_name, "wb");
    fwrite(h, sizeof(uint8_t), h_size, file);  // h_size는 헤더의 크기입니다
    fwrite(transformed_data, sizeof(uint8_t), data_size, file);
    fclose(file);
}

uint8_t* get_pcm(const char* file_path, int* data_size) {
    AVFormatContext* format_ctx = NULL;
    AVCodecContext* codec_ctx = NULL;
    AVPacket packet;
    AVFrame* frame = av_frame_alloc();
    SwrContext* swr_ctx = NULL;
    uint8_t* data = NULL;
    int data_index = 0;

    // FFmpeg 초기화
    av_register_all();
    avformat_open_input(&format_ctx, file_path, NULL, NULL);
    avformat_find_stream_info(format_ctx, NULL);

    // 오디오 스트림 찾기
    int audio_stream_index = -1;
    for (int i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_index = i;
            break;
        }
    }

    // 오디오 코덱 컨텍스트 얻기
    AVCodecParameters* codec_par = format_ctx->streams[audio_stream_index]->codecpar;

    // 오디오 코덱 찾기
    AVCodec* codec = avcodec_find_decoder(codec_par->codec_id);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }

    // 오디오 코덱 컨텍스트 초기화
    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        fprintf(stderr, "Could not allocate audio codec context\n");
        exit(1);
    }

    // 오디오 코덱 컨텍스트 설정 복사
    if (avcodec_parameters_to_context(codec_ctx, codec_par) < 0) {
        fprintf(stderr, "Could not copy codec parameters to context\n");
        exit(1);
    }

    // 오디오 코덱 열기
    if (avcodec_open2(codec_ctx, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }

    // PCM_S32LE 형식으로 리샘플링 설정
    swr_ctx = swr_alloc_set_opts(NULL, av_get_default_channel_layout(codec_par->channels),
        AV_SAMPLE_FMT_S32, codec_par->sample_rate, av_get_default_channel_layout(codec_par->channels),
        codec_ctx->sample_fmt, codec_par->sample_rate, 0, NULL);
    swr_init(swr_ctx);

    // 패킷 읽기
    while (av_read_frame(format_ctx, &packet) >= 0) {
        // 오디오 스트림이 아니면 패스
        if (packet.stream_index != audio_stream_index) {
            continue;
        }

        // 디코딩
        avcodec_send_packet(codec_ctx, &packet);
        avcodec_receive_frame(codec_ctx, frame);

        // 리샘플링
        uint8_t* buffer;
        av_samples_alloc(&buffer, NULL, codec_par->channels, frame->nb_samples, AV_SAMPLE_FMT_S32, 0);
        swr_convert(swr_ctx, &buffer, frame->nb_samples, (const uint8_t**)frame->data, frame->nb_samples);

        // 데이터 복사
        int buffer_size = av_samples_get_buffer_size(NULL, codec_par->channels, frame->nb_samples, AV_SAMPLE_FMT_S32, 0);
        data = (uint8_t*)realloc(data, data_index + buffer_size);
        memcpy(data + data_index, buffer, buffer_size);
        data_index += buffer_size;

        // 메모리 해제
        av_freep(&buffer);
    }

    // 메모리 해제
    swr_free(&swr_ctx);
    av_frame_free(&frame);

    // FFmpeg 종료
    avformat_close_input(&format_ctx);

    *data_size = data_index;
    return data;
}

typedef struct {
    int channels;
    int sample_rate;
} AudioInfo;

AudioInfo get_info(const char* file_path) {
    AVFormatContext* format_ctx = NULL;
    AVCodecParameters* codec_par = NULL;
    AudioInfo audio_info;

    // FFmpeg 초기화
    av_register_all();
    avformat_open_input(&format_ctx, file_path, NULL, NULL);
    avformat_find_stream_info(format_ctx, NULL);

    // 오디오 스트림 찾기
    int audio_stream_index = -1;
    for (int i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_index = i;
            break;
        }
    }

    // 오디오 코덱 파라미터 얻기
    codec_par = format_ctx->streams[audio_stream_index]->codecpar;

    // 채널 수와 샘플링 레이트 반환
    audio_info.channels = codec_par->channels;
    audio_info.sample_rate = codec_par->sample_rate;

    // FFmpeg 종료
    avformat_close_input(&format_ctx);

    return audio_info;
}