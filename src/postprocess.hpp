/**
 * postprocess.hpp — YOLOv8 / YOLOv26 split-output post-processing.
 * Identical model format to the reference demo (2 tensors: boxes, classes).
 */
#pragma once

#include "common.hpp"
#include <cstdint>
#include <vector>

#define OBJ_CLASS_NUM       80
#define OBJ_NUMB_MAX_SIZE   64
#define NMS_THRESH          0.45f
#define BOX_THRESH          0.25f
#define MODEL_OUT_COLS      8400    /* 80×80 + 40×40 + 20×20 */

extern const char *COCO_LABELS[OBJ_CLASS_NUM];

/**
 * Post-process split INT8 output from YOLOv8/YOLOv26 RKNN model.
 *
 * @param boxes_i8   INT8 boxes tensor  [4, 8400]  cx cy w h (model px space)
 * @param box_zp     Box quantisation zero-point
 * @param box_scale  Box quantisation scale
 * @param cls_i8     INT8 class tensor  [80, 8400] sigmoid probabilities
 * @param cls_zp     Class quantisation zero-point
 * @param cls_scale  Class quantisation scale
 * @param pad_x      Letterbox left padding (model pixels)
 * @param pad_y      Letterbox top  padding (model pixels)
 * @param scale      Letterbox scale (model_dim / orig_dim)
 * @param orig_w     Original frame width
 * @param orig_h     Original frame height
 */
std::vector<Detection> yolov8_postprocess(
    const int8_t *boxes_i8, int32_t box_zp, float box_scale,
    const int8_t *cls_i8,   int32_t cls_zp, float cls_scale,
    int pad_x, int pad_y, float scale,
    int orig_w, int orig_h);
