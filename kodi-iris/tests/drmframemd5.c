/*
 * drmframemd5 FILE DECODER
 *
 * Descodifica FILE amb DECODER demanant AV_PIX_FMT_DRM_PRIME a get_format (com
 * fa Kodi), descarrega cada fotograma a NV12 amb av_hwframe_transfer_data() i
 * escriu a stdout l'md5 de la imatge empaquetada (alineació 1), una línia per
 * fotograma: el mateix valor que la darrera columna de `-f framemd5`.
 *
 * Per stderr (log) escriu per fotograma la mida, l'AVHWFramesContext adjunt i
 * la seva mida, i una línia "HWFC CHANGE" cada cop que el decodificador
 * adjunta un context nou.  Es manté una referència a l'anterior perquè
 * l'adreça no es pugui reutilitzar.
 *
 * Existeix perquè l'eina ffmpeg amb -no_cvt_hw no reinicialitza mai el graf de
 * filtres (fftools/ffmpeg_filter.c), de manera que hwdownload no pot seguir un
 * canvi de resolució; Kodi no té aquesta limitació.
 */
#include <stdio.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/md5.h>

static int nframe;
static AVBufferRef *prev_hwfc;

static enum AVPixelFormat get_format(AVCodecContext *c, const enum AVPixelFormat *p)
{
    for (; *p != AV_PIX_FMT_NONE; p++)
        if (*p == AV_PIX_FMT_DRM_PRIME)
            return *p;
    fprintf(stderr, "drmframemd5: el decodificador no ofereix DRM_PRIME\n");
    return AV_PIX_FMT_NONE;
}

static int output(const AVFrame *f)
{
    const AVHWFramesContext *h;
    AVFrame *sw;
    uint8_t digest[16], *buf;
    int size, ret;

    if (f->format != AV_PIX_FMT_DRM_PRIME || !f->hw_frames_ctx) {
        fprintf(stderr, "drmframemd5: fotograma %d no és DRM_PRIME amb hw_frames_ctx\n", nframe);
        return AVERROR(EINVAL);
    }
    h = (const AVHWFramesContext *)f->hw_frames_ctx->data;
    if (!prev_hwfc || prev_hwfc->data != f->hw_frames_ctx->data) {
        fprintf(stderr, "HWFC CHANGE at frame %d: %p -> %p (%dx%d %s)\n", nframe,
                prev_hwfc ? (void *)prev_hwfc->data : NULL, (void *)h,
                h->width, h->height, av_get_pix_fmt_name(h->sw_format));
        av_buffer_unref(&prev_hwfc);
        prev_hwfc = av_buffer_ref(f->hw_frames_ctx);
    }
    fprintf(stderr, "frame %d %dx%d hwfc=%p %dx%d\n", nframe, f->width, f->height,
            (void *)h, h->width, h->height);

    sw = av_frame_alloc();
    if (!sw)
        return AVERROR(ENOMEM);
    sw->format = AV_PIX_FMT_NV12;
    ret = av_hwframe_transfer_data(sw, f, 0);
    if (ret < 0) {
        fprintf(stderr, "drmframemd5: transfer fotograma %d: %s\n", nframe, av_err2str(ret));
        av_frame_free(&sw);
        return ret;
    }
    size = av_image_get_buffer_size(sw->format, sw->width, sw->height, 1);
    buf = av_malloc(size);
    if (!buf) {
        av_frame_free(&sw);
        return AVERROR(ENOMEM);
    }
    av_image_copy_to_buffer(buf, size, (const uint8_t * const *)sw->data, sw->linesize,
                            sw->format, sw->width, sw->height, 1);
    av_md5_sum(digest, buf, size);
    for (int i = 0; i < 16; i++)
        printf("%02x", digest[i]);
    printf("\n");
    av_free(buf);
    av_frame_free(&sw);
    nframe++;
    return 0;
}

static int drain(AVCodecContext *c, AVFrame *f)
{
    int ret;
    while ((ret = avcodec_receive_frame(c, f)) == 0) {
        ret = output(f);
        av_frame_unref(f);
        if (ret < 0)
            return ret;
    }
    return ret;
}

int main(int argc, char **argv)
{
    AVFormatContext *fc = NULL;
    AVCodecContext *c;
    const AVCodec *codec;
    AVPacket *pkt;
    AVFrame *f;
    int si, ret;

    if (argc != 3) {
        fprintf(stderr, "ús: %s FITXER DECODIFICADOR\n", argv[0]);
        return 2;
    }
    av_log_set_level(AV_LOG_DEBUG);
    if (avformat_open_input(&fc, argv[1], NULL, NULL) < 0 ||
        avformat_find_stream_info(fc, NULL) < 0)
        return 1;
    si = av_find_best_stream(fc, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    codec = avcodec_find_decoder_by_name(argv[2]);
    if (si < 0 || !codec)
        return 1;
    c = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(c, fc->streams[si]->codecpar);
    c->pkt_timebase = fc->streams[si]->time_base;
    c->get_format = get_format;
    if ((ret = avcodec_open2(c, codec, NULL)) < 0) {
        fprintf(stderr, "drmframemd5: avcodec_open2: %s\n", av_err2str(ret));
        return 1;
    }

    pkt = av_packet_alloc();
    f = av_frame_alloc();
    while (av_read_frame(fc, pkt) >= 0) {
        if (pkt->stream_index == si) {
            while ((ret = avcodec_send_packet(c, pkt)) == AVERROR(EAGAIN)) {
                ret = drain(c, f);
                if (ret < 0 && ret != AVERROR(EAGAIN))
                    goto end;
            }
            if (ret < 0)
                goto end;
            ret = drain(c, f);
            if (ret < 0 && ret != AVERROR(EAGAIN))
                goto end;
        }
        av_packet_unref(pkt);
    }
    avcodec_send_packet(c, NULL);
    ret = drain(c, f);
end:
    av_buffer_unref(&prev_hwfc);
    av_packet_free(&pkt);
    av_frame_free(&f);
    avcodec_free_context(&c);
    avformat_close_input(&fc);
    if (ret == AVERROR_EOF)
        ret = 0;
    if (ret < 0)
        fprintf(stderr, "drmframemd5: %s\n", av_err2str(ret));
    return ret < 0;
}
