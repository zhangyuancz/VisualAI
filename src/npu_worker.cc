/**
 * npu_worker.cc — RGA zero-copy scale + RKNN multi-core inference.
 *
 * Zero-copy chain (no CPU memcpy):
 *   [RKMPP DMA fd / NV12] ──RGA──▶ [RKNN DMA input / RGB888 640×640]
 *                                          │
 *                                     rknn_run()
 *                                          │
 *                             [INT8 output bufs, direct virt_addr access]
 */
#include "npu_worker.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>

extern "C" {
#include <libavutil/hwcontext_drm.h>
#include <libdrm/drm_fourcc.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

#include "mjpeg_server.hpp"
#include <sys/mman.h>

/* ── SoC detection ───────────────────────────────────────────────────────────── */
int npu_core_count()
{
    FILE *f = fopen("/proc/device-tree/compatible", "rb");
    if (!f) return 1;
    char buf[256] = {};
    fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    /* compatible is a list of null-separated strings; scan each one */
    for (int i = 0; i < (int)sizeof(buf); ) {
        const char *s = buf + i;
        if (strstr(s, "rk3588")) return 3;
        if (strstr(s, "rk3576")) return 2;
        int len = (int)strlen(s);
        if (len == 0) break;
        i += len + 1;
    }
    return 1;
}

/* ── Construction / destruction ─────────────────────────────────────────────── */
NpuWorker::NpuWorker(int id,
                     BoundedQueue<AVFrame *>    &in_queue,
                     BoundedQueue<InferResult>  &out_queue)
    : id_(id), in_queue_(in_queue), out_queue_(out_queue) {}

NpuWorker::~NpuWorker() {
    stop();

    if (rga_dst_handle_) releasebuffer_handle(rga_dst_handle_);

    for (auto *m : out_mems_) if (m) rknn_destroy_mem(ctx_, m);
    if (in_mem_)  rknn_destroy_mem(ctx_, in_mem_);
    if (ctx_)     rknn_destroy(ctx_);

    /* Web encoder cleanup */
    if (jpeg_ctx_)     avcodec_free_context(&jpeg_ctx_);
    if (sws_nv12_rgb_) sws_freeContext(sws_nv12_rgb_);
    if (sws_rgb_yuv_)  sws_freeContext(sws_rgb_yuv_);
}

/* ── init ──────────────────────────────────────────────────────────────────── */
bool NpuWorker::init(const NpuWorkerConfig &cfg)
{
    model_w_ = cfg.model_w;
    model_h_ = cfg.model_h;

    /* Load model file */
    FILE *fp = fopen(cfg.model_path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "[NpuWorker %d] Cannot open model: %s\n",
                id_, cfg.model_path.c_str());
        return false;
    }
    fseek(fp, 0, SEEK_END);
    long   msize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> mbuf(msize);
    fread(mbuf.data(), 1, msize, fp);
    fclose(fp);

    /* Init RKNN context */
    int ret = rknn_init(&ctx_, mbuf.data(), msize, 0, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[NpuWorker %d] rknn_init failed: %d\n", id_, ret);
        return false;
    }

    /* Bind to NPU core */
    rknn_set_core_mask(ctx_, cfg.core_mask);
    fprintf(stderr, "[NpuWorker %d] core_mask=0x%x\n", id_, (unsigned)cfg.core_mask);

    /* Query I/O counts */
    rknn_input_output_num io_num{};
    rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (io_num.n_input < 1 || io_num.n_output < 2) {
        fprintf(stderr, "[NpuWorker %d] Unexpected I/O count: in=%u out=%u\n",
                id_, io_num.n_input, io_num.n_output);
        return false;
    }
    n_output_ = io_num.n_output;

    /* ── Input tensor: allocate DMA mem, set zero-copy io_mem ── */
    in_attr_.index = 0;
    rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr_, sizeof(in_attr_));
    /* Allocate RKNN-managed DMA buffer sized for UINT8 NHWC [1,H,W,3] */
    in_mem_ = rknn_create_mem(ctx_, model_w_ * model_h_ * 3);
    if (!in_mem_) {
        fprintf(stderr, "[NpuWorker %d] rknn_create_mem (input) failed\n", id_);
        return false;
    }
    /* pass_through=0: RKNN converts UINT8 RGB to INT8 internally */
    in_attr_.pass_through = 0;
    in_attr_.type         = RKNN_TENSOR_UINT8;
    in_attr_.fmt          = RKNN_TENSOR_NHWC;
    ret = rknn_set_io_mem(ctx_, in_mem_, &in_attr_);
    if (ret < 0) {
        fprintf(stderr, "[NpuWorker %d] rknn_set_io_mem (input) failed: %d\n", id_, ret);
        return false;
    }

    /* Import RKNN input DMA fd into RGA as the destination buffer */
    rga_dst_handle_ = importbuffer_fd(in_mem_->fd, model_w_ * model_h_ * 3);
    if (!rga_dst_handle_) {
        fprintf(stderr, "[NpuWorker %d] importbuffer_fd (dst) failed\n", id_);
        return false;
    }
    rga_dst_ = wrapbuffer_handle(rga_dst_handle_, model_w_, model_h_,
                                 RK_FORMAT_RGB_888);

    /* ── Output tensors: allocate DMA mems, set zero-copy io_mem ── */
    out_mems_.resize(n_output_, nullptr);
    out_attrs_.resize(n_output_);
    for (uint32_t i = 0; i < n_output_; ++i) {
        out_attrs_[i].index = i;
        rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attrs_[i], sizeof(rknn_tensor_attr));

        out_mems_[i] = rknn_create_mem(ctx_, out_attrs_[i].size_with_stride);
        if (!out_mems_[i]) {
            fprintf(stderr, "[NpuWorker %d] rknn_create_mem (output %u) failed\n", id_, i);
            return false;
        }
        out_attrs_[i].pass_through = 0;
        ret = rknn_set_io_mem(ctx_, out_mems_[i], &out_attrs_[i]);
        if (ret < 0) {
            fprintf(stderr, "[NpuWorker %d] rknn_set_io_mem (output %u) failed: %d\n",
                    id_, i, ret);
            return false;
        }
    }

    fprintf(stderr, "[NpuWorker %d] init OK  model=%s  inputs=%u  outputs=%u\n",
            id_, cfg.model_path.c_str(), io_num.n_input, io_num.n_output);
    return true;
}

/* ── start / stop ──────────────────────────────────────────────────────────── */
void NpuWorker::start() {
    running_.store(true);
    stop_req_.store(false);
    thread_ = std::thread(&NpuWorker::run, this);
    if (mjpeg_server_)
        web_thread_ = std::thread(&NpuWorker::web_run, this);
}

void NpuWorker::stop() {
    stop_req_.store(true);
    web_queue_.close();
    if (web_thread_.joinable()) web_thread_.join();
    if (thread_.joinable())     thread_.join();
    running_.store(false);
}

/* ── RGA scale: NV12 DRM frame → RGB888 640×640 letterbox (zero-copy) ──────── */
bool NpuWorker::rga_scale(AVFrame *frame, int &pad_x, int &pad_y, float &scale)
{
    if (frame->format != AV_PIX_FMT_DRM_PRIME) {
        fprintf(stderr, "[NpuWorker %d] Frame is not DRM_PRIME (fmt=%d); "
                        "zero-copy unavailable\n", id_, frame->format);
        return false;
    }

    auto *drm = reinterpret_cast<AVDRMFrameDescriptor *>(frame->data[0]);
    if (!drm || drm->nb_objects < 1) return false;

    int src_fd   = drm->objects[0].fd;
    int src_w    = frame->width;
    int src_h    = frame->height;

    /* Detect format (NV12 vs NV21 etc.) */
    uint32_t fourcc = drm->layers[0].format;
    RgaSURF_FORMAT rga_src_fmt =
        (fourcc == DRM_FORMAT_NV21) ? RK_FORMAT_YCrCb_420_SP
                                    : RK_FORMAT_YCbCr_420_SP;  /* NV12 default */

    /* Luma stride from DRM plane pitch (bytes = pixels for YUV 8-bit) */
    int y_stride = (drm->nb_layers > 0 && drm->layers[0].nb_planes > 0)
                       ? static_cast<int>(drm->layers[0].planes[0].pitch)
                       : src_w;
    int h_stride = src_h;  /* MPP usually aligns to 16, but use actual height */

    /* Letterbox math */
    scale = std::min(static_cast<float>(model_w_) / src_w,
                     static_cast<float>(model_h_) / src_h);
    int new_w = static_cast<int>(src_w * scale + 0.5f);
    int new_h = static_cast<int>(src_h * scale + 0.5f);
    pad_x = (model_w_ - new_w) / 2;
    pad_y = (model_h_ - new_h) / 2;

    /* Cache resolution to detect changes */
    if (src_w != orig_w_ || src_h != orig_h_) {
        orig_w_ = src_w;
        orig_h_ = src_h;
    }

    /* Import source DMA fd into RGA (released at end of function) */
    rga_buffer_handle_t src_handle =
        importbuffer_fd(src_fd, src_w, src_h, rga_src_fmt);
    if (!src_handle) {
        fprintf(stderr, "[NpuWorker %d] importbuffer_fd (src) failed\n", id_);
        return false;
    }
    rga_buffer_t rga_src =
        wrapbuffer_handle_t(src_handle, src_w, src_h, y_stride, h_stride, rga_src_fmt);

    /* Pass 1: fill 640×640 destination with gray (letterbox padding) */
    im_rect full_rect = {0, 0, model_w_, model_h_};
    imfill(rga_dst_, full_rect, 0x808080);

    /* Pass 2: scale NV12→RGB + letterbox into centre sub-region */
    im_rect src_rect = {0, 0, src_w, src_h};
    im_rect dst_rect = {pad_x, pad_y, new_w, new_h};
    IM_STATUS st = improcess(rga_src, rga_dst_, {},
                             src_rect, dst_rect, {},
                             IM_SYNC);

    releasebuffer_handle(src_handle);

    if (st != IM_STATUS_SUCCESS) {
        fprintf(stderr, "[NpuWorker %d] improcess failed: %d\n", id_, (int)st);
        return false;
    }
    return true;
}

/* ── Web debug helpers ─────────────────────────────────────────────────────── */

/* Deterministic bright color per class */
static void class_color(int cls_id, uint8_t &r, uint8_t &g, uint8_t &b)
{
    static const uint8_t palette[][3] = {
        {255, 56,  56 }, {255,157,151}, {255,112, 31}, {255,178, 29},
        {207,210, 49 }, { 72,249, 10}, {146,204, 23}, { 61,219,134},
        { 26,147, 52 }, {  0,212,187}, { 44,153,168}, {  0,194,255},
        { 52, 69,147 }, {100,115,255}, {  0, 24,236}, {132, 56,255},
        { 82,  0,133 }, {203, 56,255}, {255,149,200}, {255, 55,199},
    };
    int idx = ((cls_id % 20) + 20) % 20;
    r = palette[idx][0]; g = palette[idx][1]; b = palette[idx][2];
}

/* Draw filled rectangle outline (thickness px) on RGB24 image */
static void draw_rect_rgb(uint8_t *rgb, int img_w, int img_h,
                          int x1, int y1, int x2, int y2,
                          uint8_t r, uint8_t g, uint8_t b, int thickness = 2)
{
    x1 = std::max(0, x1); y1 = std::max(0, y1);
    x2 = std::min(img_w - 1, x2); y2 = std::min(img_h - 1, y2);

    auto setpix = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= img_w || y >= img_h) return;
        uint8_t *p = rgb + (y * img_w + x) * 3;
        p[0] = r; p[1] = g; p[2] = b;
    };

    for (int x = x1; x <= x2; ++x)
        for (int t = 0; t < thickness; ++t) {
            setpix(x, y1 + t);
            setpix(x, y2 - t);
        }
    for (int y = y1; y <= y2; ++y)
        for (int t = 0; t < thickness; ++t) {
            setpix(x1 + t, y);
            setpix(x2 - t, y);
        }
}

bool NpuWorker::init_web_encoder(int w, int h)
{
    sws_nv12_rgb_ = sws_getContext(w, h, AV_PIX_FMT_NV12,
                                   w, h, AV_PIX_FMT_RGB24,
                                   SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_nv12_rgb_) {
        fprintf(stderr, "[NpuWorker %d] sws NV12->RGB failed\n", id_);
        return false;
    }

    sws_rgb_yuv_ = sws_getContext(w, h, AV_PIX_FMT_RGB24,
                                  w, h, AV_PIX_FMT_YUVJ420P,
                                  SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_rgb_yuv_) {
        fprintf(stderr, "[NpuWorker %d] sws RGB->YUV failed\n", id_);
        return false;
    }

    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!codec) {
        fprintf(stderr, "[NpuWorker %d] MJPEG encoder not found\n", id_);
        return false;
    }

    jpeg_ctx_                 = avcodec_alloc_context3(codec);
    jpeg_ctx_->width          = w;
    jpeg_ctx_->height         = h;
    jpeg_ctx_->pix_fmt        = AV_PIX_FMT_YUVJ420P;
    jpeg_ctx_->time_base      = {1, 30};
    jpeg_ctx_->flags         |= AV_CODEC_FLAG_QSCALE;
    jpeg_ctx_->global_quality = FF_QP2LAMBDA * 7;   /* ~85% quality */

    if (avcodec_open2(jpeg_ctx_, codec, nullptr) < 0) {
        avcodec_free_context(&jpeg_ctx_);
        fprintf(stderr, "[NpuWorker %d] avcodec_open2 (MJPEG) failed\n", id_);
        return false;
    }

    web_rgb_buf_.resize(static_cast<size_t>(w * h * 3));
    web_cap_w_ = w;
    web_cap_h_ = h;

    fprintf(stderr, "[NpuWorker %d] Web encoder ready: %dx%d MJPEG q=85\n",
            id_, w, h);
    return true;
}

void NpuWorker::encode_and_push_web(const uint8_t *nv12, int w, int h,
                                    int y_stride,
                                    const std::vector<Detection> &dets)
{
    /* Lazy init on first frame */
    if (!jpeg_ctx_) {
        if (!init_web_encoder(w, h)) return;
    }

    /* ── NV12 → RGB24 ── */
    const uint8_t *src_planes[4] = {
        nv12,
        nv12 + y_stride * h,  /* UV plane starts after Y plane */
        nullptr, nullptr
    };
    const int src_strides[4] = { y_stride, y_stride, 0, 0 };

    uint8_t *dst_planes[4] = { web_rgb_buf_.data(), nullptr, nullptr, nullptr };
    int      dst_strides[4] = { w * 3, 0, 0, 0 };

    sws_scale(sws_nv12_rgb_, src_planes, src_strides, 0, h,
              dst_planes, dst_strides);

    /* ── Draw bounding boxes ── */
    for (const auto &d : dets) {
        uint8_t r, g, b;
        class_color(d.cls_id, r, g, b);
        draw_rect_rgb(web_rgb_buf_.data(), w, h,
                      static_cast<int>(d.box.left),
                      static_cast<int>(d.box.top),
                      static_cast<int>(d.box.right),
                      static_cast<int>(d.box.bottom),
                      r, g, b, 3);
    }

    /* ── RGB24 → YUVJ420P → JPEG ── */
    AVFrame *yuv = av_frame_alloc();
    yuv->format  = AV_PIX_FMT_YUVJ420P;
    yuv->width   = w;
    yuv->height  = h;
    av_frame_get_buffer(yuv, 1);

    const uint8_t *rgb_planes[4] = { web_rgb_buf_.data(), nullptr, nullptr, nullptr };
    const int      rgb_strides[4] = { w * 3, 0, 0, 0 };
    sws_scale(sws_rgb_yuv_, rgb_planes, rgb_strides, 0, h,
              yuv->data, yuv->linesize);

    yuv->pts = web_pts_++;
    avcodec_send_frame(jpeg_ctx_, yuv);
    av_frame_free(&yuv);

    AVPacket *pkt = av_packet_alloc();
    if (avcodec_receive_packet(jpeg_ctx_, pkt) == 0) {
        std::vector<uint8_t> jpeg(pkt->data, pkt->data + pkt->size);
        mjpeg_server_->push_frame(std::move(jpeg));
    }
    av_packet_free(&pkt);
}

/* ── web_run: dedicated MJPEG encoder thread ────────────────────────────────── */
void NpuWorker::web_run()
{
    WebTask task;
    while (web_queue_.pop(task)) {
        encode_and_push_web(task.nv12.data(), task.w, task.h,
                            task.y_stride, task.dets);
    }
}

void NpuWorker::run()
{
    AVFrame *frame = nullptr;

    while (!stop_req_.load()) {
        if (!in_queue_.pop(frame)) break;  /* queue closed */

        InferResult result;
        result.frame_id     = frame->pts;
        result.npu_worker_id = id_;

        /* ── RGA scale (zero-copy NV12→RGB 640×640) ── */
        int   pad_x = 0, pad_y = 0;
        float scale = 1.0f;
        int64_t t0 = now_us();
        bool ok = rga_scale(frame, pad_x, pad_y, scale);
        result.scale_us = now_us() - t0;

        int orig_w = frame->width;
        int orig_h = frame->height;

        /* ── Capture NV12 for web debug (before freeing the DMA frame) ── */
        WebTask web_task;
        bool    web_ok = false;
        if (mjpeg_server_ && ok) {
            auto *drm = reinterpret_cast<AVDRMFrameDescriptor *>(frame->data[0]);
            if (drm && drm->nb_objects > 0) {
                int    dma_fd = drm->objects[0].fd;
                size_t dma_sz = static_cast<size_t>(drm->objects[0].size);
                web_task.y_stride = (drm->nb_layers > 0 && drm->layers[0].nb_planes > 0)
                                        ? static_cast<int>(drm->layers[0].planes[0].pitch)
                                        : orig_w;
                void *mapped = mmap(nullptr, dma_sz, PROT_READ, MAP_SHARED, dma_fd, 0);
                if (mapped != MAP_FAILED) {
                    web_task.nv12.resize(dma_sz);
                    memcpy(web_task.nv12.data(), mapped, dma_sz);
                    munmap(mapped, dma_sz);
                    web_task.w   = orig_w;
                    web_task.h   = orig_h;
                    web_task.pts = frame->pts;
                    web_ok = true;
                }
            }
        }

        av_frame_free(&frame);  /* release DMA buffer back to RKMPP pool */

        if (!ok) continue;

        /* ── RKNN inference (input already in DMA, no copy) ── */
        t0 = now_us();
        int ret = rknn_run(ctx_, nullptr);
        result.infer_us = now_us() - t0;

        if (ret < 0) {
            fprintf(stderr, "[NpuWorker %d] rknn_run failed: %d\n", id_, ret);
            continue;
        }

        /* ── Post-process (direct INT8 virt_addr access, no rknn_outputs_get) ── */
        t0 = now_us();

        const auto &box_attr = out_attrs_[0];
        const auto &cls_attr = out_attrs_[1];
        const auto *boxes_i8 = static_cast<const int8_t *>(out_mems_[0]->virt_addr);
        const auto *cls_i8   = static_cast<const int8_t *>(out_mems_[1]->virt_addr);

        result.dets = yolov8_postprocess(
            boxes_i8, box_attr.zp, box_attr.scale,
            cls_i8,   cls_attr.zp, cls_attr.scale,
            pad_x, pad_y, scale,
            orig_w, orig_h);

        result.postproc_us = now_us() - t0;

        /* ── Web debug: hand off to encoder thread (non-blocking) ── */
        if (mjpeg_server_ && web_ok) {
            web_task.dets = result.dets;
            web_queue_.push_latest(std::move(web_task), [](WebTask &) {});
        }

        out_queue_.push(std::move(result));
    }

    running_.store(false);
}
