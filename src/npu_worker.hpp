/**
 * npu_worker.hpp — RGA letterbox scale + RKNN inference worker.
 *
 * Zero-copy pipeline:
 *   RKMPP DMA fd (NV12) → RGA (scale + YUV→RGB) → RKNN input DMA buffer
 *   → RKNN inference → INT8 outputs read directly from pre-allocated mems
 *
 * Multiple workers can run concurrently, each pinned to a different NPU core.
 *
 * Optional web debug: call set_mjpeg_server() before start().
 * Worker will capture, annotate and JPEG-encode each frame for the browser.
 */
#pragma once

#include "common.hpp"
#include "postprocess.hpp"

#include <atomic>
#include <thread>
#include <vector>

#include <rknn_api.h>
#include <rga/im2d.h>
#include <rga/im2d_buffer.h>

extern "C" {
#include <libavutil/frame.h>
}

/* Forward declarations to avoid pulling heavy headers into every TU */
class  MjpegServer;
struct SwsContext;
struct AVCodecContext;

/* NPU core assignments indexed by worker_id */
static constexpr rknn_core_mask kCoreMap[] = {
    RKNN_NPU_CORE_0,
    RKNN_NPU_CORE_1,
    RKNN_NPU_CORE_2,
};

struct NpuWorkerConfig {
    int         worker_id      = 0;
    int         num_workers    = 1;   /* total workers → chooses core mask */
    std::string model_path;
    int         model_w        = 640;
    int         model_h        = 640;
};

class NpuWorker {
public:
    NpuWorker(int id, BoundedQueue<AVFrame *> &in_queue,
              BoundedQueue<InferResult> &out_queue);
    ~NpuWorker();

    bool init(const NpuWorkerConfig &cfg);
    void start();
    void stop();

    bool is_running() const { return running_.load(); }

    /* Set MJPEG server for web debug visualization (optional, call before start) */
    void set_mjpeg_server(MjpegServer *srv) { mjpeg_server_ = srv; }

private:
    void run();

    /* RGA: letterbox scale NV12 frame → RGB888 640×640 (zero-copy) */
    bool rga_scale(AVFrame *frame, int &pad_x, int &pad_y, float &scale);

    /* Web debug: NV12 → RGB → draw boxes → JPEG → push to MjpegServer */
    bool init_web_encoder(int w, int h);
    void encode_and_push_web(const uint8_t *nv12, int w, int h, int y_stride,
                             const std::vector<Detection> &dets);

    /* ── Members ── */
    int                       id_;
    BoundedQueue<AVFrame *>  &in_queue_;
    BoundedQueue<InferResult> &out_queue_;

    /* RKNN */
    rknn_context              ctx_      = 0;
    rknn_tensor_mem          *in_mem_   = nullptr;
    uint32_t                  n_output_ = 0;
    std::vector<rknn_tensor_mem *> out_mems_;
    std::vector<rknn_tensor_attr>  out_attrs_;
    rknn_tensor_attr          in_attr_{};

    /* RGA */
    rga_buffer_handle_t       rga_dst_handle_ = 0;
    rga_buffer_t              rga_dst_{};

    /* Config */
    int   model_w_ = 640;
    int   model_h_ = 640;
    int   orig_w_  = 0;
    int   orig_h_  = 0;

    /* Web debug visualization */
    MjpegServer    *mjpeg_server_  = nullptr;
    SwsContext     *sws_nv12_rgb_  = nullptr;   /* NV12 → RGB24 */
    SwsContext     *sws_rgb_yuv_   = nullptr;   /* RGB24 → YUVJ420P */
    AVCodecContext *jpeg_ctx_      = nullptr;
    std::vector<uint8_t> web_nv12_buf_;         /* raw NV12 capture */
    std::vector<uint8_t> web_rgb_buf_;          /* RGB24 intermediate */
    int64_t         web_pts_       = 0;
    int             web_cap_w_     = 0;
    int             web_cap_h_     = 0;

    std::thread               thread_;
    std::atomic<bool>         running_{false};
    std::atomic<bool>         stop_req_{false};
};
