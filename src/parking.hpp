/**
 * parking.hpp — Parking spot occupancy detection.
 *
 * ParkingSpot corners are in ORIGINAL video resolution.
 * Overlap check and quad drawing both operate in original resolution directly
 * (postprocess.cc already de-letterboxes detections to original coords).
 *
 * Tools:
 *   tools/annotate_spots.py — interactive corner annotation GUI
 *   parking_spots.conf        — spot definition (plain text, one line per spot)
 */

#pragma once

#include "common.hpp"

#include <cstdint>
#include <string>
#include <vector>

/**
 * ParkingSpot — user-supplied corner definitions, in original video resolution.
 * Corners are ordered clockwise or counterclockwise.
 */
struct ParkingSpot {
    int    id;               /* sequential index */
    char   label[32];       /* user-visible name (e.g. "A1", "B2") */
    float  corners[4][2];   /* [4 corners][x,y] — original resolution */
};

/**
 * load_parking_config(path) — parse a parking_spots.conf file.
 * Returns an empty vector on error (caller should print error and exit).
 * Expected format (one line per spot):
 *   id  label  x1  y1  x2  y2  x3  y3  x4  y4   (integer pixel coords)
 * Lines starting with # are comments.
 */
std::vector<ParkingSpot> load_parking_config(const std::string &path);

/**
 * check_spot_occupancy — for each parking spot, check if any car detection
 * overlaps it by at least `overlap_pct` of the spot's area.
 *
 * @param spots      parking spot definitions (original resolution)
 * @param dets       YOLOv8 detections (ALREADY in original resolution)
 * @param pad_x      letterbox horizontal padding in model input
 * @param pad_y      letterbox vertical padding in model input
 * @param scale      model_scale = min(640/src_w, 640/src_h)
 * @param orig_w     original frame width
 * @param orig_h     original frame height
 * @param thresh     overlap fraction threshold (e.g. 0.25 = 25%%)
 * @return one OccupancyResult per spot, in the same order as `spots`
 */
std::vector<OccupancyResult>
check_spot_occupancy(const std::vector<ParkingSpot> &spots,
                     const std::vector<struct Detection> &dets,
                     int pad_x, int pad_y, float scale,
                     int orig_w, int orig_h,
                     float thresh = 0.25f);

/**
 * draw_quad_rgb — draw a 4-corner polygon outline on an RGB24 buffer.
 *
 * @param rgb       pointer to RGB24 pixel data
 * @param img_w     image width in pixels
 * @param img_h     image height in pixels
 * @param corners   4 corners [4][x,y] in img coordinates
 * @param r, g, b   RGB color
 * @param thickness line thickness in pixels
 */
void draw_quad_rgb(uint8_t *rgb, int img_w, int img_h,
                   const float corners[4][2],
                   uint8_t r, uint8_t g, uint8_t b,
                   int thickness = 2);
