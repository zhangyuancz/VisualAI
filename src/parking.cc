/**
 * parking.cc — Parking spot occupancy detection implementation.
 */

#include "parking.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

/* ── Geometry helpers ───────────────────────────────────────────────────────── */

static float poly_area(const float poly[][2], int n) {
    float area = 0.0f;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += poly[i][0] * poly[j][1];
        area -= poly[j][0] * poly[i][1];
    }
    return std::abs(area) * 0.5f;
}

/**
 * Clip `poly` (up to 8 vertices) against one axis-aligned edge:
 *   keep vertices where (clip_b.x - clip_a.x)*(p.y - clip_a.y) >=
 *                      (clip_b.y - clip_a.y)*(p.x - clip_a.x)
 * Returns number of output vertices (fills up to 8 entries).
 */
static int clip_edge(const float in[][2], int n_in,
                     float out[][2],
                     float ax, float ay, float bx, float by)
{
    int n_out = 0;
    for (int i = 0; i < n_in; ++i) {
        const float *cur  = in[i];
        const float *nxt  = in[(i + 1) % n_in];

        auto inside = [&](const float *p) {
            return (bx - ax) * (p[1] - ay) >= (by - ay) * (p[0] - ax);
        };

        bool cur_in = inside(cur);
        bool nxt_in = inside(nxt);

        if (cur_in) {
            out[n_out][0] = cur[0];
            out[n_out][1] = cur[1];
            ++n_out;

            if (!nxt_in) {
                // Line cur→nxt exits: add intersection
                float dx = nxt[0] - cur[0];
                float dy = nxt[1] - cur[1];
                float denom = (bx - ax) * dy - (by - ay) * dx;
                if (std::abs(denom) > 1e-9f) {
                    float t = ((ax - cur[0]) * dy - (ay - cur[1]) * dx) / denom;
                    t = std::max(0.0f, std::min(1.0f, t));
                    out[n_out][0] = cur[0] + t * dx;
                    out[n_out][1] = cur[1] + t * dy;
                    ++n_out;
                }
            }
        } else if (nxt_in) {
            // Line cur→nxt enters: add intersection
            float dx = nxt[0] - cur[0];
            float dy = nxt[1] - cur[1];
            float denom = (bx - ax) * dy - (by - ay) * dx;
            if (std::abs(denom) > 1e-9f) {
                float t = ((ax - cur[0]) * dy - (ay - cur[1]) * dx) / denom;
                t = std::max(0.0f, std::min(1.0f, t));
                out[n_out][0] = cur[0] + t * dx;
                out[n_out][1] = cur[1] + t * dy;
                ++n_out;
            }
        }
    }
    return n_out;
}

/**
 * Sutherland-Hodgman polygon clipping of `quad` (4 vertices) against
 * the axis-aligned bounding box of `box`.
 * Returns the clipped polygon area (or 0 if no intersection).
 */
static float aabb_quad_intersection_area(const float quad[4][2],
                                         float bx1, float by1,
                                         float bx2, float by2)
{
    if (bx2 <= bx1 || by2 <= by1) return 0.0f;

    float poly[8][2] = {
        { quad[0][0], quad[0][1] },
        { quad[1][0], quad[1][1] },
        { quad[2][0], quad[2][1] },
        { quad[3][0], quad[3][1] },
    };
    int n = 4;
    float tmp[8][2];

    // Clip against 4 edges of the AABB (clockwise traversal so "left of edge" = inside).
    // Each directed edge keeps vertices on the interior side of that AABB boundary.
    n = clip_edge(poly, n, tmp, bx1, by1, bx2, by1); if (n < 3) return 0.0f;  // top:    keep y >= by1
    for (int i = 0; i < n; ++i) { poly[i][0] = tmp[i][0]; poly[i][1] = tmp[i][1]; }
    n = clip_edge(poly, n, tmp, bx2, by1, bx2, by2); if (n < 3) return 0.0f;  // right:  keep x <= bx2
    for (int i = 0; i < n; ++i) { poly[i][0] = tmp[i][0]; poly[i][1] = tmp[i][1]; }
    n = clip_edge(poly, n, tmp, bx2, by2, bx1, by2); if (n < 3) return 0.0f;  // bottom: keep y <= by2
    for (int i = 0; i < n; ++i) { poly[i][0] = tmp[i][0]; poly[i][1] = tmp[i][1]; }
    n = clip_edge(poly, n, tmp, bx1, by2, bx1, by1); if (n < 3) return 0.0f;  // left:   keep x >= bx1

    return poly_area(tmp, n);
}

/* ── Coordinate transform ───────────────────────────────────────────────────── */

/**
 * Map corners from original resolution to model resolution.
 * Inverse letterbox transform:
 *   x_model = (x_orig - pad_x) / scale
 *   y_model = (y_orig - pad_y) / scale
 */
static void map_corners_to_model(const float orig[4][2],
                                 int pad_x, int pad_y, float scale,
                                 float mapped[4][2])
{
    for (int i = 0; i < 4; ++i) {
        mapped[i][0] = (orig[i][0] - static_cast<float>(pad_x)) / scale;
        mapped[i][1] = (orig[i][1] - static_cast<float>(pad_y)) / scale;
    }
}

/* ── Config loader ──────────────────────────────────────────────────────────── */

std::vector<ParkingSpot> load_parking_config(const std::string &path)
{
    std::vector<ParkingSpot> spots;
    FILE *f = fopen(path.c_str(), "r");
    if (!f) {
        fprintf(stderr, "[parking] Cannot open config: %s\n", path.c_str());
        return {};
    }

    char line[512];
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        ++lineno;

        // Strip leading/trailing whitespace
        char *s = line;
        while (*s == ' ' || *s == '\t') ++s;
        if (!*s || s[0] == '#' || s[0] == '\n')
            continue;

        int id;
        char label[32];
        int x[4], y[4];
        int n = sscanf(s, "%d %31s %d %d %d %d %d %d %d %d",
                       &id, label, &x[0], &y[0], &x[1], &y[1],
                       &x[2], &y[2], &x[3], &y[3]);
        if (n < 10) {
            fprintf(stderr, "[parking] line %d: expected 10 fields, got %d: %s",
                    lineno, n, line);
            continue;
        }

        ParkingSpot spot{};
        spot.id = id;
        snprintf(spot.label, sizeof(spot.label), "%s", label);
        for (int i = 0; i < 4; ++i) {
            spot.corners[i][0] = static_cast<float>(x[i]);
            spot.corners[i][1] = static_cast<float>(y[i]);
        }
        spots.push_back(spot);
    }

    fclose(f);
    if (spots.empty())
        fprintf(stderr, "[parking] No valid spots found in: %s\n", path.c_str());
    else
        fprintf(stderr, "[parking] Loaded %zu spots from: %s\n", spots.size(), path.c_str());

    return spots;
}

/* ── Occupancy check ────────────────────────────────────────────────────────── */

std::vector<OccupancyResult>
check_spot_occupancy(const std::vector<ParkingSpot> &spots,
                     const std::vector<Detection>   &dets,
                     int pad_x, int pad_y, float scale,
                     int /*orig_w*/, int /*orig_h*/,
                     float thresh)
{
    (void)scale; (void)pad_x; (void)pad_y;  /* currently unused — detections already de-letterboxed */

    std::vector<OccupancyResult> results;
    results.reserve(spots.size());

    for (const auto &spot : spots) {
        // Compute quad area
        float quad_area = poly_area(spot.corners, 4);
        if (quad_area <= 0.0f) {
            OccupancyResult r{};
            r.spot_id = spot.id;
            strncpy(r.label, spot.label, sizeof(r.label) - 1);
            r.occupied = false;
            r.overlap_pct = 0.0f;
            results.push_back(r);
            continue;
        }

        // Map corners to model space for overlay (MJPEG draw uses orig coords directly)
        float mapped[4][2];
        map_corners_to_model(spot.corners, pad_x, pad_y, scale, mapped);

        // Find the best (largest) overlap among all detections
        float best_overlap = 0.0f;
        for (const auto &det : dets) {
            // Only consider "car" class (COCO class_id == 2)
            if (det.cls_id != 2) continue;

            const float bx1 = det.box.left;
            const float by1 = det.box.top;
            const float bx2 = det.box.right;
            const float by2 = det.box.bottom;

            // Quick AABB reject
            float qsx = std::min({ spot.corners[0][0], spot.corners[1][0],
                                   spot.corners[2][0], spot.corners[3][0] });
            float qsy = std::min({ spot.corners[0][1], spot.corners[1][1],
                                   spot.corners[2][1], spot.corners[3][1] });
            float qsx2 = std::max({ spot.corners[0][0], spot.corners[1][0],
                                    spot.corners[2][0], spot.corners[3][0] });
            float qsy2 = std::max({ spot.corners[0][1], spot.corners[1][1],
                                    spot.corners[2][1], spot.corners[3][1] });
            if (bx2 <= qsx || by2 <= qsy || bx1 >= qsx2 || by1 >= qsy2)
                continue;  // AABB disjoint

            float inter = aabb_quad_intersection_area(spot.corners, bx1, by1, bx2, by2);
            float ratio = inter / quad_area;
            if (ratio > best_overlap)
                best_overlap = ratio;
        }

        OccupancyResult r{};
        r.spot_id = spot.id;
        snprintf(r.label, sizeof(r.label), "%s", spot.label);
        r.occupied    = (best_overlap >= thresh);
        r.overlap_pct = best_overlap;
        for (int i = 0; i < 4; ++i) {
            r.mapped_corners[i][0] = mapped[i][0];
            r.mapped_corners[i][1] = mapped[i][1];
        }
        results.push_back(r);
    }

    return results;
}

/* ── Quad drawing on RGB24 ──────────────────────────────────────────────────── */

/**
 * Draw a 4-corner polygon outline on an RGB24 image.
 * Clipping: all corners are clamped to [0, img_w-1] × [0, img_h-1].
 */
void draw_quad_rgb(uint8_t *rgb, int img_w, int img_h,
                   const float corners_f[4][2],
                   uint8_t r, uint8_t g, uint8_t b,
                   int thickness)
{
    // Convert to integers with clamping
    int cx[4], cy[4];
    for (int i = 0; i < 4; ++i) {
        cx[i] = std::max(0, std::min(img_w - 1, static_cast<int>(corners_f[i][0])));
        cy[i] = std::min(img_h - 1, static_cast<int>(corners_f[i][1]));
    }

    auto setpix = [&](int x, int y) {
        if (x < 0 || x >= img_w || y < 0 || y >= img_h) return;
        uint8_t *p = rgb + (y * img_w + x) * 3;
        p[0] = r; p[1] = g; p[2] = b;
    };

    for (int e = 0; e < 4; ++e) {
        int x1 = cx[e], y1 = cy[e];
        int x2 = cx[(e + 1) % 4], y2 = cy[(e + 1) % 4];

        // Bresenham line
        int dx = std::abs(x2 - x1);
        int dy = std::abs(y2 - y1);
        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;
        int err = dx - dy;

        while (true) {
            for (int t = 0; t < thickness; ++t) {
                setpix(x1 - t, y1); setpix(x1 + t, y1);
                setpix(x1, y1 - t); setpix(x1, y1 + t);
            }
            if (x1 == x2 && y1 == y2) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x1 += sx; }
            if (e2 < dx)  { err += dx; y1 += sy; }
        }
    }
}
