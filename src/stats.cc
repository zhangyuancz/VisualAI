/**
 * stats.cc — Stats tracker implementation.
 */
#include "stats.hpp"
#include <cstring>

StatsTracker::StatsTracker() {
    start_us_    = now_us();
    last_snap_us_ = start_us_;
    getrusage(RUSAGE_SELF, &ru_last_);
}

void StatsTracker::record(const InferResult &r) {
    ++total_frames_;
    ++frames_in_interval_;
    total_dets_      += static_cast<int64_t>(r.dets.size());
    sum_scale_us_    += r.scale_us;
    sum_infer_us_    += r.infer_us;
    sum_postproc_us_ += r.postproc_us;
    ++n_infer_;
}

bool StatsTracker::snapshot(double interval_s, StatsSnapshot &out) {
    int64_t now = now_us();
    double elapsed_since_snap = (now - last_snap_us_) / 1e6;
    if (elapsed_since_snap < interval_s) return false;

    double total_elapsed = (now - start_us_) / 1e6;

    /* CPU usage over last interval */
    struct rusage ru_now;
    getrusage(RUSAGE_SELF, &ru_now);
    double cpu_user = (ru_now.ru_utime.tv_sec  - ru_last_.ru_utime.tv_sec)  * 1e6 +
                      (ru_now.ru_utime.tv_usec - ru_last_.ru_utime.tv_usec);
    double cpu_sys  = (ru_now.ru_stime.tv_sec  - ru_last_.ru_stime.tv_sec)  * 1e6 +
                      (ru_now.ru_stime.tv_usec - ru_last_.ru_stime.tv_usec);
    double cpu_pct  = (cpu_user + cpu_sys) / (elapsed_since_snap * 1e6) * 100.0;
    ru_last_ = ru_now;

    out.wall_s        = total_elapsed;
    out.inst_fps      = frames_in_interval_ / elapsed_since_snap;
    out.avg_fps       = total_elapsed > 0 ? total_frames_ / total_elapsed : 0.0;
    out.avg_scale_ms  = n_infer_ > 0 ? sum_scale_us_    / n_infer_ / 1000.0 : 0.0;
    out.avg_infer_ms  = n_infer_ > 0 ? sum_infer_us_    / n_infer_ / 1000.0 : 0.0;
    out.avg_postproc_ms = n_infer_ > 0 ? sum_postproc_us_ / n_infer_ / 1000.0 : 0.0;
    out.cpu_pct       = cpu_pct;
    out.total_frames  = total_frames_;
    out.total_dets    = total_dets_;

    /* Reset interval counters */
    last_snap_us_        = now;
    frames_in_interval_  = 0;
    sum_scale_us_        = 0.0;
    sum_infer_us_        = 0.0;
    sum_postproc_us_     = 0.0;
    n_infer_             = 0;

    return true;
}
