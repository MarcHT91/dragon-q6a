/* primefallback FILE PRESET(0|1): obre el descodificador HEVC/H.264 de programari com ho fa
 * el camí DRMPRIME de Kodi (pix_fmt=DRM_PRIME prefixat) i diu si avcodec_open2 l'accepta. */
#include <stdio.h>
#include <stdlib.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
int main(int argc, char **argv)
{
    AVFormatContext *fc = NULL;
    if (argc < 3 || avformat_open_input(&fc, argv[1], NULL, NULL) < 0 ||
        avformat_find_stream_info(fc, NULL) < 0)
        return 2;
    int si = av_find_best_stream(fc, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    const AVCodec *cd = avcodec_find_decoder(fc->streams[si]->codecpar->codec_id);
    AVCodecContext *c = avcodec_alloc_context3(cd);
    AVCodecParameters *p = fc->streams[si]->codecpar;
    c->extradata = av_mallocz(p->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy(c->extradata, p->extradata, p->extradata_size);
    c->extradata_size = p->extradata_size;
    if (atoi(argv[2]))
        c->pix_fmt = AV_PIX_FMT_DRM_PRIME;
    int ret = avcodec_open2(c, cd, NULL);
    printf("%s preset=%s -> %s\n", cd->name, argv[2], ret < 0 ? "REFUSAT" : "OBERT");
    avcodec_free_context(&c);
    avformat_close_input(&fc);
    return ret < 0;
}
