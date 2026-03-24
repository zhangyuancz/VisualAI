/**
 * decoder.cc — RTSP demux + RKMPP hardware decode.
 *
 * Zero-copy path: h264_rkmpp / hevc_rkmpp decoder outputs frames as
 * AV_PIX_FMT_DRM_PRIME (DMA buf fd in AVDRMFrameDescriptor).
 */
#include "decoder.hpp"

#include <algorithm>
#include <cstring>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/time.h>
}

/* Map codec id → RKMPP decoder name */
static const char *rkmpp_decoder_name(enum AVCodecID id) {
    switch (id) {
    case AV_CODEC_ID_H264:       return "h264_rkmpp";
    case AV_CODEC_ID_HEVC:       return "hevc_rkmpp";
    case AV_CODEC_ID_VP8:        return "vp8_rkmpp";
    case AV_CODEC_ID_VP9:        return "vp9_rkmpp";
    case AV_CODEC_ID_AV1:        return "av1_rkmpp";
    case AV_CODEC_ID_MPEG2VIDEO: return "mpeg2_rkmpp";
    case AV_CODEC_ID_MPEG4:      return "mpeg4_rkmpp";
    default:                      return nullptr;
    }
}

/* ── Construction / destruction ─────────────────────────────────────────────── */
Decoder::Decoder(BoundedQueue<AVFrame *> &out_queue) : out_queue_(out_queue) {}

Decoder::~Decoder() {
    stop();
    close_stream();
    av_buffer_unref(&hw_dev_);
}

/* ── get_format callback: prefer DRM_PRIME for zero-copy ───────────────────── */
enum AVPixelFormat Decoder::get_hw_format(AVCodecContext * /*ctx*/,
                                          const enum AVPixelFormat *pix_fmts)
{
    for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == AV_PIX_FMT_DRM_PRIME) return *p;
    }
    /* Fallback (should not happen with rkmpp decoder + hw_device_ctx) */
    return pix_fmts[0];
}

/* ── interrupt callback: abort blocking FFmpeg calls on stop or read stall ──── */
int Decoder::interrupt_cb(void *opaque)
{
    auto *d = static_cast<Decoder *>(opaque);
    if (d->stop_req_.load()) return 1;
    if (d->cfg_.read_timeout_s > 0) {
        int64_t elapsed_us = now_us() - d->last_pkt_us_.load();
        if (elapsed_us > static_cast<int64_t>(d->cfg_.read_timeout_s) * 1'000'000LL)
            return 1;
    }
    return 0;
}

/* ── open_stream: (re)open format + codec context ──────────────────────────── */
bool Decoder::open_stream(VideoInfo &info)
{
    char errbuf[256];
    int  ret;

    /* Reset watchdog FIRST so interrupt_cb doesn't fire during open/probe */
    last_pkt_us_.store(now_us());

    fprintf(stderr, "[Decoder] Opening stream: %s\n", cfg_.url.c_str());

    /* Pre-allocate context so interrupt_cb can be attached before avformat_open_input,
     * enabling stop_req_ to abort a blocking connect attempt. */
    fmt_ctx_ = avformat_alloc_context();
    fmt_ctx_->interrupt_callback.callback = Decoder::interrupt_cb;
    fmt_ctx_->interrupt_callback.opaque   = this;

    AVDictionary *opts = nullptr;
    av_dict_set(&opts, "rtsp_transport",  cfg_.rtsp_transport.c_str(), 0);
    av_dict_set(&opts, "stimeout",        "5000000",  0);
    av_dict_set(&opts, "analyzeduration", "1000000",  0);
    av_dict_set(&opts, "probesize",       "1000000",  0);
    av_dict_set(&opts, "max_delay",       "500000",   0);

    ret = avformat_open_input(&fmt_ctx_, cfg_.url.c_str(), nullptr, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        fprintf(stderr, "[Decoder] avformat_open_input failed: %s\n", errbuf);
        fmt_ctx_ = nullptr;  /* avformat_open_input frees ctx on failure */
        return false;
    }

    /* Refresh watchdog after connection is established */
    last_pkt_us_.store(now_us());

    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        fprintf(stderr, "[Decoder] avformat_find_stream_info failed: %s\n", errbuf);
        avformat_close_input(&fmt_ctx_);
        return false;
    }

    int video_idx = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    int audio_idx = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (video_idx < 0) {
        fprintf(stderr, "[Decoder] No video stream found\n");
        avformat_close_input(&fmt_ctx_);
        return false;
    }
    vid_idx_ = video_idx;

    AVStream *vs  = fmt_ctx_->streams[video_idx];
    auto     *vcp = vs->codecpar;

    info.width            = vcp->width;
    info.height           = vcp->height;
    info.bitrate          = vcp->bit_rate;
    info.video_stream_idx = video_idx;
    info.audio_stream_idx = audio_idx;
    info.codec_name       = avcodec_get_name(vcp->codec_id);
    info.fps = (vs->avg_frame_rate.den && vs->avg_frame_rate.num)
                   ? static_cast<float>(av_q2d(vs->avg_frame_rate))
                   : static_cast<float>(av_q2d(vs->r_frame_rate));

    /* Find hardware decoder */
    const AVCodec *decoder = nullptr;
    const char    *hw_name = rkmpp_decoder_name(vcp->codec_id);
    if (hw_name) {
        decoder = avcodec_find_decoder_by_name(hw_name);
        if (!decoder)
            fprintf(stderr, "[Decoder] HW decoder '%s' not found; trying software\n", hw_name);
    }
    if (!decoder) {
        decoder = avcodec_find_decoder(vcp->codec_id);
        if (!decoder) {
            fprintf(stderr, "[Decoder] No decoder for %s\n", info.codec_name.c_str());
            avformat_close_input(&fmt_ctx_);
            return false;
        }
    }

    dec_ctx_ = avcodec_alloc_context3(decoder);
    if (!dec_ctx_) {
        fprintf(stderr, "[Decoder] alloc context failed\n");
        avformat_close_input(&fmt_ctx_);
        return false;
    }

    avcodec_parameters_to_context(dec_ctx_, vcp);
    dec_ctx_->time_base = vs->time_base;

    if (hw_dev_) {
        dec_ctx_->hw_device_ctx  = av_buffer_ref(hw_dev_);
        dec_ctx_->get_format     = get_hw_format;
        dec_ctx_->extra_hw_frames = cfg_.num_hw_frames;
    }

    dec_ctx_->flags  |= AV_CODEC_FLAG_LOW_DELAY;
    dec_ctx_->flags2 |= AV_CODEC_FLAG2_FAST;

    AVDictionary *codec_opts = nullptr;
    if (hw_dev_) av_dict_set(&codec_opts, "fast_mode", "1", 0);

    ret = avcodec_open2(dec_ctx_, decoder, &codec_opts);
    av_dict_free(&codec_opts);
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        fprintf(stderr, "[Decoder] avcodec_open2: %s\n", errbuf);
        avcodec_free_context(&dec_ctx_);
        avformat_close_input(&fmt_ctx_);
        return false;
    }

    last_pkt_us_.store(now_us());
    fprintf(stderr, "[Decoder] Stream ready: %dx%d  %.2f fps  codec=%s  hw=%s\n",
            info.width, info.height, info.fps, decoder->name,
            hw_dev_ ? "RKMPP" : "SW");
    return true;
}

/* ── close_stream: release format + codec context ──────────────────────────── */
void Decoder::close_stream()
{
    avcodec_free_context(&dec_ctx_);
    avformat_close_input(&fmt_ctx_);
}

/* ── init ──────────────────────────────────────────────────────────────────── */
bool Decoder::init(const DecoderConfig &cfg, VideoInfo &info)
{
    cfg_ = cfg;

    /* Create RKMPP HW device context once; reused across reconnects */
    char errbuf[256];
    int  ret = av_hwdevice_ctx_create(&hw_dev_, AV_HWDEVICE_TYPE_RKMPP, nullptr, nullptr, 0);
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        fprintf(stderr, "[Decoder] av_hwdevice_ctx_create: %s\n", errbuf);
        hw_dev_ = nullptr;
    }

    return open_stream(info);
}

/* ── start / stop ──────────────────────────────────────────────────────────── */
void Decoder::start() {
    running_.store(true);
    stop_req_.store(false);
    thread_ = std::thread(&Decoder::run, this);
}

void Decoder::stop() {
    stop_req_.store(true);
    if (thread_.joinable()) thread_.join();
    running_.store(false);
}

/* ── decode thread ─────────────────────────────────────────────────────────── */
void Decoder::run()
{
    AVPacket *pkt   = av_packet_alloc();
    AVFrame  *frame = av_frame_alloc();

    int retries  = 0;
    int max_ret  = cfg_.reconnect_retries;   /* -1 = unlimited */

    fprintf(stderr, "[Decoder] Decode thread started (reconnect=%s max=%d timeout=%ds)\n",
            max_ret != 0 ? "on" : "off", max_ret, cfg_.read_timeout_s);

    /* Outer loop: reconnect on network errors */
    while (!stop_req_.load()) {
        bool need_reconnect = false;
        bool queue_closed   = false;

        /* ── Inner read/decode loop ── */
        while (!stop_req_.load() && !queue_closed) {
            /* fmt_ctx_ can be null when a previous open_stream() attempt failed */
            if (!fmt_ctx_) { need_reconnect = true; break; }

            int ret = av_read_frame(fmt_ctx_, pkt);

            if (ret == AVERROR(EAGAIN)) { av_usleep(1000); continue; }

            if (ret == AVERROR_EXIT) {
                /* Triggered by interrupt_cb: either explicit stop or read stall */
                if (!stop_req_.load()) {
                    fprintf(stderr, "[Decoder] Read stall: no packet for %d s; will reconnect\n",
                            cfg_.read_timeout_s);
                    need_reconnect = true;
                }
                break;
            }

            if (ret == AVERROR_EOF) {
                /* Stream ended (sender stopped pushing). Treat as reconnectable. */
                fprintf(stderr, "[Decoder] Stream EOF (sender disconnected); will reconnect\n");
                need_reconnect = true;
                break;
            }

            if (ret < 0) {
                char errbuf[256];
                av_strerror(ret, errbuf, sizeof(errbuf));
                fprintf(stderr, "[Decoder] Read error: %s; will reconnect\n", errbuf);
                need_reconnect = true;
                break;
            }

            last_pkt_us_.store(now_us());  /* reset watchdog on every packet */

            if (pkt->stream_index != vid_idx_) { av_packet_unref(pkt); continue; }

            ret = avcodec_send_packet(dec_ctx_, pkt);
            av_packet_unref(pkt);
            if (ret < 0 && ret != AVERROR(EAGAIN)) continue;

            while (!stop_req_.load()) {
                ret = avcodec_receive_frame(dec_ctx_, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) break;

                frame->pts = frame_id_++;

                AVFrame *queued = av_frame_clone(frame);
                av_frame_unref(frame);

                if (queued) {
                    bool ok = out_queue_.push_latest(queued, [this](AVFrame *f) {
                        av_frame_free(&f);
                        ++dropped_;
                    });
                    if (!ok) { queue_closed = true; break; }
                }
            }
        }

        if (stop_req_.load() || queue_closed) break;
        if (!need_reconnect) break;  /* should not happen but guard anyway */

        /* ── Reconnect with exponential backoff ── */
        if (max_ret == 0) {
            fprintf(stderr, "[Decoder] Reconnect disabled; exiting\n");
            break;
        }
        if (max_ret > 0 && retries >= max_ret) {
            fprintf(stderr, "[Decoder] Max reconnect attempts (%d) reached; exiting\n", max_ret);
            break;
        }

        int delay_ms = std::min(cfg_.reconnect_delay_ms * (1 << std::min(retries, 5)), 30000);
        const char *limit_str = (max_ret < 0) ? "unlimited" : "";
        int limit_num = (max_ret > 0) ? max_ret : 0;
        if (max_ret < 0)
            fprintf(stderr, "[Decoder] Reconnecting in %d ms (attempt %d, %s)...\n",
                    delay_ms, retries + 1, limit_str);
        else
            fprintf(stderr, "[Decoder] Reconnecting in %d ms (attempt %d/%d)...\n",
                    delay_ms, retries + 1, limit_num);

        /* Interruptible sleep so stop() wakes us quickly */
        for (int elapsed = 0; elapsed < delay_ms && !stop_req_.load(); elapsed += 50)
            av_usleep(50 * 1000);

        if (stop_req_.load()) break;

        close_stream();
        VideoInfo dummy{};
        if (!open_stream(dummy)) {
            fprintf(stderr, "[Decoder] Reconnect attempt %d failed; will retry\n", retries + 1);
            ++retries;
            continue;
        }

        fprintf(stderr, "[Decoder] Reconnected successfully (was attempt %d)\n", retries + 1);
        retries = 0;
    }

    /* Flush decoder before exiting */
    if (!stop_req_.load() && dec_ctx_) {
        avcodec_send_packet(dec_ctx_, nullptr);
        while (avcodec_receive_frame(dec_ctx_, frame) >= 0) av_frame_unref(frame);
    }

    fprintf(stderr, "[Decoder] Decode thread exiting (frames=%lld dropped=%lld)\n",
            (long long)frame_id_, (long long)dropped_.load());

    av_frame_free(&frame);
    av_packet_free(&pkt);
    out_queue_.close();
    running_.store(false);
}
