/**
 * decoder.hpp — RTSP demux + RKMPP hardware decode thread.
 *
 * Produces AVFrame* objects in DRM_PRIME format (DMA fd accessible via
 * AVDRMFrameDescriptor) and posts them to a BoundedQueue for zero-copy
 * downstream processing.
 */
#pragma once

#include "common.hpp"

#include <atomic>
#include <string>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
}

struct DecoderConfig {
    std::string url;
    std::string rtsp_transport = "tcp";   /* "tcp" | "udp" */
    int         num_hw_frames  = 16;      /* RKMPP frame pool size */
};

struct VideoInfo {
    int         width, height;
    float       fps;
    int64_t     bitrate;
    std::string codec_name;
    int         video_stream_idx;
    int         audio_stream_idx;
};

class Decoder {
public:
    explicit Decoder(BoundedQueue<AVFrame *> &out_queue);
    ~Decoder();

    /* Initialise codec; fills info. Returns false on error. */
    bool init(const DecoderConfig &cfg, VideoInfo &info);

    /* Start decode thread. */
    void start();

    /* Request stop and join. */
    void stop();

    bool is_running() const { return running_.load(); }
    int64_t dropped_frames() const { return dropped_.load(); }

private:
    void run();

    static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
                                            const enum AVPixelFormat *pix_fmts);

    /* FFmpeg interrupt callback — returns 1 to abort blocking calls when stopping */
    static int interrupt_cb(void *opaque) {
        return static_cast<Decoder *>(opaque)->stop_req_.load() ? 1 : 0;
    }

    BoundedQueue<AVFrame *> &out_queue_;

    AVFormatContext *fmt_ctx_  = nullptr;
    AVCodecContext  *dec_ctx_  = nullptr;
    AVBufferRef     *hw_dev_   = nullptr;
    int              vid_idx_  = -1;
    int64_t          frame_id_ = 0;

    std::thread      thread_;
    std::atomic<bool>    running_{false};
    std::atomic<bool>    stop_req_{false};
    std::atomic<int64_t> dropped_{0};
};
