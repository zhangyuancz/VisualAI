/**
 * stats.hpp — Latency / throughput / CPU usage tracker.
 */
#pragma once

#include "common.hpp"
#include <sys/resource.h>
#include <cstdint>

struct StatsSnapshot {
    double   wall_s;
    double   inst_fps;       /* FPS in last interval */
    double   avg_fps;        /* FPS since start */
    double   avg_scale_ms;   /* RGA latency */
    double   avg_infer_ms;   /* RKNN latency */
    double   avg_postproc_ms;
    double   cpu_pct;        /* process CPU % over last interval */
    int64_t  total_frames;
    int64_t  total_dets;
};

class StatsTracker {
public:
    StatsTracker();

    /* Call once per InferResult */
    void record(const InferResult &r);

    /* Returns snapshot if interval elapsed; returns false otherwise */
    bool snapshot(double interval_s, StatsSnapshot &out);

private:
    int64_t       start_us_;
    int64_t       last_snap_us_;
    int64_t       total_frames_   = 0;
    int64_t       total_dets_     = 0;
    int64_t       frames_in_interval_ = 0;

    double        sum_scale_us_   = 0.0;
    double        sum_infer_us_   = 0.0;
    double        sum_postproc_us_ = 0.0;
    int64_t       n_infer_        = 0;

    struct rusage ru_last_;
};
