/* seektest FILE DECODER N: imita els salts de Kodi (CDVDVideoCodecDRMPRIME::Reset):
 * descodifica, buida amb un paquet NULL fins a EOF, avcodec_flush_buffers, salta i continua.
 * Mesura els ms fins al primer fotograma després de cada salt. Surt 1 si algun > 1000 ms o falla. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
static enum AVPixelFormat gf(AVCodecContext *c, const enum AVPixelFormat *p)
{ for (; *p != AV_PIX_FMT_NONE; p++) if (*p == AV_PIX_FMT_DRM_PRIME) return *p; return AV_PIX_FMT_NONE; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "ús: %s FITXER DESCODIFICADOR N\n", argv[0]); return 2; }
    if (getenv("SEEK_VERBOSE")) av_log_set_level(AV_LOG_DEBUG);
    AVFormatContext *fc = NULL;
    if (avformat_open_input(&fc, argv[1], NULL, NULL) < 0 || avformat_find_stream_info(fc, NULL) < 0) return 2;
    int si = av_find_best_stream(fc, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    AVStream *st = fc->streams[si];
    const AVCodec *cd = avcodec_find_decoder_by_name(argv[2]);
    AVCodecContext *c = avcodec_alloc_context3(cd);
    avcodec_parameters_to_context(c, st->codecpar);
    c->get_format = gf; c->pix_fmt = AV_PIX_FMT_DRM_PRIME;
    if (avcodec_open2(c, cd, NULL) < 0) return 2;
    AVPacket *p = av_packet_alloc(); AVFrame *f = av_frame_alloc();
    int n = atoi(argv[3]), rc = 0;
    double dur = fc->duration > 0 ? fc->duration / 1e6 : 60;
    for (int i = 0; i <= n; i++) {
        if (i > 0) {
            /* Reset() de Kodi */
            int r = avcodec_send_packet(c, NULL);
            if (r < 0 && r != AVERROR_EOF) printf("  drain send: %s\n", av_err2str(r));
            int drained = 0;
            for (;;) { r = avcodec_receive_frame(c, f); if (r == AVERROR_EOF) break;
                       if (r < 0) { printf("  drain recv: %s\n", av_err2str(r)); break; } drained++; av_frame_unref(f); }
            avcodec_flush_buffers(c);
            double tgt = (i % 2) ? dur * 0.7 : dur * 0.2;
            tgt += i;  /* posicions diferents a cada salt */
            av_seek_frame(fc, -1, (int64_t)(tgt * AV_TIME_BASE), AVSEEK_FLAG_BACKWARD);
            printf("salt %d a %.0fs (buidats %d)", i, tgt, drained);
        } else printf("inici");
        double t0 = now(); int got = 0, errs = 0;
        while (!got && now() - t0 < 5000 && av_read_frame(fc, p) >= 0) {
            if (p->stream_index == si) {
                int r = avcodec_send_packet(c, p);
                if (r < 0 && r != AVERROR(EAGAIN)) { errs++; if (errs < 4) printf(" [send %s]", av_err2str(r)); }
                while (avcodec_receive_frame(c, f) == 0) { got++; av_frame_unref(f); }
            }
            av_packet_unref(p);
        }
        double dt = now() - t0;
        printf(" -> %s en %.0f ms%s\n", got ? "fotograma" : "CAP fotograma", dt, errs ? " (errors)" : "");
        if (!got || dt > 1000 || errs) rc = 1;
        /* descodifica ~1 s més per tenir el pipeline ple abans del salt següent */
        for (int k = 0; k < 24 && av_read_frame(fc, p) >= 0; ) {
            if (p->stream_index == si) { avcodec_send_packet(c, p); while (avcodec_receive_frame(c, f) == 0) { av_frame_unref(f); k++; } }
            av_packet_unref(p);
        }
    }
    av_frame_free(&f); av_packet_free(&p); avcodec_free_context(&c); avformat_close_input(&fc);
    return rc;
}
