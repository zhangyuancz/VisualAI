/**
 * postprocess.cc — YOLOv8 / YOLOv26 post-processing (NMS + coord mapping)
 *
 * Optimised two-pass scan:
 *   Pass 1: scan cls tensor row-by-row (c outer, j inner) — sequential memory
 *           access, cache-friendly. Track per-anchor INT8 max without any float ops.
 *   Pass 2: only dequantise + decode anchors that passed the INT8 threshold.
 *           Avoids 8400×80 float conversions when most anchors are below threshold.
 */
#include "postprocess.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>

const char *COCO_LABELS[OBJ_CLASS_NUM] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static float iou(const Detection &a, const Detection &b) {
    float ix1 = std::max(a.box.left,   b.box.left);
    float iy1 = std::max(a.box.top,    b.box.top);
    float ix2 = std::min(a.box.right,  b.box.right);
    float iy2 = std::min(a.box.bottom, b.box.bottom);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float ua = (a.box.right - a.box.left) * (a.box.bottom - a.box.top)
             + (b.box.right - b.box.left) * (b.box.bottom - b.box.top) - inter;
    return ua <= 0.f ? 0.f : inter / ua;
}

static void nms(std::vector<Detection> &dets, float thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection &a, const Detection &b) { return a.conf > b.conf; });
    std::vector<bool> sup(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (sup[i]) continue;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!sup[j] && dets[i].cls_id == dets[j].cls_id &&
                iou(dets[i], dets[j]) > thresh)
                sup[j] = true;
        }
    }
    std::vector<Detection> kept;
    kept.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i)
        if (!sup[i]) kept.push_back(dets[i]);
    dets = std::move(kept);
}

/* 线程局部缓存，避免每帧堆分配 / thread-local cache to avoid per-frame heap alloc */
static thread_local int8_t  tl_max_raw[MODEL_OUT_COLS];
static thread_local int16_t tl_best_cls[MODEL_OUT_COLS];

std::vector<Detection> yolov8_postprocess(
    const int8_t *boxes_i8, int32_t box_zp, float box_scale,
    const int8_t *cls_i8,   int32_t cls_zp, float cls_scale,
    int pad_x, int pad_y, float scale,
    int orig_w, int orig_h)
{
    const int N = MODEL_OUT_COLS;

    /* ── 第一遍：按行顺序扫描，找每个 anchor 的最大 INT8 类分数 ──────────
     * cls_i8 布局 [OBJ_CLASS_NUM][N]，c 为行，j 为列。
     * 外层 c、内层 j 保证顺序内存访问，缓存友好。                          */
    for (int j = 0; j < N; ++j) tl_max_raw[j] = INT8_MIN;
    for (int j = 0; j < N; ++j) tl_best_cls[j] = 0;

    for (int c = 0; c < OBJ_CLASS_NUM; ++c) {
        const int8_t *row = cls_i8 + c * N;
        for (int j = 0; j < N; ++j) {
            if (row[j] > tl_max_raw[j]) {
                tl_max_raw[j]  = row[j];
                tl_best_cls[j] = static_cast<int16_t>(c);
            }
        }
    }

    /* 将置信度阈值转换到 INT8 空间，避免在第二遍做浮点比较 */
    const int8_t thresh_raw = static_cast<int8_t>(
        std::max(-128.f, std::min(127.f, BOX_THRESH / cls_scale + cls_zp)));

    /* ── 第二遍：只对通过阈值的 anchor 做浮点反量化和坐标解码 ─────────── */
    std::vector<Detection> dets;
    dets.reserve(64);

    for (int j = 0; j < N; ++j) {
        if (tl_max_raw[j] <= thresh_raw) continue;

        float prob = (static_cast<float>(tl_max_raw[j]) - cls_zp) * cls_scale;
        if (prob < BOX_THRESH) continue;   /* 浮点二次确认 */

        int cls = tl_best_cls[j];

        /* 反量化 cx,cy,w,h（模型像素空间） */
        float cx = (static_cast<float>(boxes_i8[0 * N + j]) - box_zp) * box_scale;
        float cy = (static_cast<float>(boxes_i8[1 * N + j]) - box_zp) * box_scale;
        float bw = (static_cast<float>(boxes_i8[2 * N + j]) - box_zp) * box_scale;
        float bh = (static_cast<float>(boxes_i8[3 * N + j]) - box_zp) * box_scale;

        /* 去除 letterbox 偏移，映射回原图坐标 */
        float x1 = (cx - bw * 0.5f - pad_x) / scale;
        float y1 = (cy - bh * 0.5f - pad_y) / scale;
        float x2 = (cx + bw * 0.5f - pad_x) / scale;
        float y2 = (cy + bh * 0.5f - pad_y) / scale;

        x1 = clampf(x1, 0.f, static_cast<float>(orig_w - 1));
        y1 = clampf(y1, 0.f, static_cast<float>(orig_h - 1));
        x2 = clampf(x2, 0.f, static_cast<float>(orig_w - 1));
        y2 = clampf(y2, 0.f, static_cast<float>(orig_h - 1));
        if (x2 <= x1 || y2 <= y1) continue;

        Detection d;
        d.cls_id = cls;
        d.conf   = prob;
        d.box    = {x1, y1, x2, y2};
        strncpy(d.label, COCO_LABELS[cls], sizeof(d.label) - 1);
        d.label[sizeof(d.label) - 1] = '\0';
        dets.push_back(d);
    }

    nms(dets, NMS_THRESH);
    if (dets.size() > OBJ_NUMB_MAX_SIZE) dets.resize(OBJ_NUMB_MAX_SIZE);
    return dets;
}
