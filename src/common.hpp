/**
 * common.hpp — Shared types and thread-safe bounded queue.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/frame.h>
}

/* ── Detection result ──────────────────────────────────────────────────────── */
struct BBox {
    float left, top, right, bottom;
};

struct Detection {
    char  label[64];
    int   cls_id;
    float conf;
    BBox  box;
};

/* ── Parking occupancy result ──────────────────────────────────────────────── */

/**
 * OccupancyResult — per-spot occupancy for the MJPEG overlay and logging.
 * Corners are stored in MODEL resolution so the overlay draw code doesn't
 * need to recompute the letterbox transform.
 */
struct OccupancyResult {
    int    spot_id;
    char   label[32];
    bool   occupied;
    float  overlap_pct;       /* intersection / quad_area */
    float  mapped_corners[4][2]; /* corners in MODEL resolution */
};

struct InferResult {
    int64_t              frame_id    = 0;
    int64_t              decode_us   = 0;
    int64_t              scale_us    = 0;
    int64_t              infer_us    = 0;
    int64_t              postproc_us = 0;
    int                  npu_worker_id;
    std::vector<Detection>      dets;
    /* Letterbox params for coordinate transforms */
    int                  orig_w  = 0, orig_h  = 0;
    int                  pad_x   = 0, pad_y   = 0;
    float                scale   = 1.0f;
    /* Parking occupancy (filled when parking config is loaded) */
    std::vector<OccupancyResult> occupancy;
};

/* ── Thread-safe bounded queue ─────────────────────────────────────────────── */
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : cap_(capacity), closed_(false) {}

    /* Push item; blocks if full. Returns false if queue was closed. */
    bool push(T item) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_push_.wait(lk, [this] { return q_.size() < cap_ || closed_; });
        if (closed_) return false;
        q_.push(std::move(item));
        cv_pop_.notify_one();
        return true;
    }

    /* Real-time push: never blocks. If full, evicts the oldest item first.
     * on_evict(T) is called outside the lock for each evicted item.
     * Returns false if the queue is closed (item is also passed to on_evict). */
    template <typename EvictFn>
    bool push_latest(T item, EvictFn on_evict) {
        T    evicted{};
        bool did_evict = false;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (closed_) { on_evict(item); return false; }
            if (q_.size() >= cap_) {
                evicted   = std::move(q_.front());
                q_.pop();
                did_evict = true;
            }
            q_.push(std::move(item));
            cv_pop_.notify_one();
        }
        if (did_evict) on_evict(evicted);   /* free DMA ref outside lock */
        return true;
    }

    /* Pop item; blocks if empty. Returns false if queue closed and empty. */
    bool pop(T &item) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_pop_.wait(lk, [this] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        item = std::move(q_.front());
        q_.pop();
        cv_push_.notify_one();
        return true;
    }

    /* Pop with timeout. Returns false on timeout OR queue closed+empty.
     * Lets callers re-check external stop flags (e.g. g_running) periodically. */
    bool pop_timeout(T &item, int timeout_ms) {
        std::unique_lock<std::mutex> lk(mtx_);
        bool ready = cv_pop_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                                      [this] { return !q_.empty() || closed_; });
        if (!ready || q_.empty()) return false;
        item = std::move(q_.front());
        q_.pop();
        cv_push_.notify_one();
        return true;
    }

    /* Signal that no more items will be pushed. Unblocks waiting consumers. */
    void close() {
        std::lock_guard<std::mutex> lk(mtx_);
        closed_ = true;
        cv_pop_.notify_all();
        cv_push_.notify_all();
    }

    bool is_closed() {
        std::lock_guard<std::mutex> lk(mtx_);
        return closed_;
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mtx_);
        return q_.size();
    }

private:
    std::queue<T>           q_;
    std::mutex              mtx_;
    std::condition_variable cv_push_, cv_pop_;
    size_t                  cap_;
    bool                    closed_;
};

/* ── Timestamp helper ──────────────────────────────────────────────────────── */
static inline int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}
