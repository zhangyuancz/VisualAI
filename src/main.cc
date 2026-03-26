/**
 * main.cc — RTSP → RKMPP decode → RGA scale → RKNN YOLOv8 inference
 *
 * Zero-copy pipeline on RK3576 / RK3588:
 *   [RTSP] → RKMPP decode → frame_queue → RGA scale → RKNN infer → postprocess → output
 *
 * Usage:
 *   rtsp_yolo <rtsp_url> <model.rknn> [options]
 *
 * Options:
 *   --transport tcp|udp     RTSP transport (default: tcp)
 *   --npu -1|0|1|2          NPU core spec (default: -1 = auto)
 *   --interval S            Stats print interval in seconds (default: 3)
 *   --verbose               Also print per-interval RGA/NPU/postproc timing
 */

#include "common.hpp"
#include "decoder.hpp"
#include "npu_worker.hpp"
#include "stats.hpp"
#include "mjpeg_server.hpp"
#include "parking.hpp"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <sys/resource.h>

static volatile int g_running = 1;
static void on_signal(int) { g_running = 0; }

static void print_usage(const char *prog) {
    printf("Usage: %s <rtsp_url> <model.rknn> [options]\n", prog);
    printf("Options:\n");
    printf("  --transport tcp|udp   RTSP transport (default: tcp)\n");
    printf("  --npu -1|0|1|2        NPU core spec (default: -1)\n");
    printf("                          -1 = auto: one thread per core (SoC-adaptive)\n");
    printf("                           0 = pin to NPU core 0 (1 thread)\n");
    printf("                           1 = pin to NPU core 1 (1 thread)\n");
    printf("                           2 = pin to NPU core 2 (1 thread, RK3588 only)\n");
    printf("  --interval S          Stats interval in seconds (default: 3)\n");
    printf("  --verbose             Print per-interval RGA/NPU/postproc timing\n");
    printf("  --web-port N          Enable MJPEG debug viewer on http://<ip>:N\n");
    printf("  --spots-config FILE   Parking spot config (from annotate_spots.py)\n");
    printf("\nExample:\n");
    printf("  %s rtsp://192.168.1.100:8554/stream model.rknn --npu -1\n\n", prog);
}

static void print_video_info(const VideoInfo &info) {
    printf("\n=== Stream Info ===\n");
    printf("  Codec     : %s\n", info.codec_name.c_str());
    printf("  Resolution: %dx%d\n", info.width, info.height);
    printf("  FPS       : %.2f\n", info.fps);
    printf("  Bitrate   : %lld kbps\n", (long long)(info.bitrate / 1000));
    printf("===================\n\n");
}

static void print_stats(const StatsSnapshot &s, int num_workers, bool verbose,
                        int64_t dropped) {
    printf("[%6.1f s]  fps(inst/avg)=%5.1f/%5.1f  cpu=%5.1f%%  "
           "dets/frame=%.1f  drop=%lld",
           s.wall_s, s.inst_fps, s.avg_fps, s.cpu_pct,
           s.total_frames > 0 ? static_cast<double>(s.total_dets) / s.total_frames : 0.0,
           (long long)dropped);
    if (verbose) {
        printf("  rga=%.2fms  npu=%.2fms  post=%.2fms  workers=%d",
               s.avg_scale_ms, s.avg_infer_ms, s.avg_postproc_ms, num_workers);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    if (argc < 3) { print_usage(argv[0]); return 1; }

    std::string rtsp_url   = argv[1];
    std::string model_path = argv[2];

    std::string transport   = "tcp";
    int         core_spec   = -1;   /* -1 = auto (all cores), 0/1/2 = pin */
    double      interval_s  = 3.0;
    bool        verbose     = false;
    int         web_port    = 0;
    std::string parking_config_path;

    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--transport") && i + 1 < argc) transport = argv[++i];
        else if (!strcmp(argv[i], "--npu")       && i + 1 < argc) core_spec   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--interval") && i + 1 < argc) interval_s  = atof(argv[++i]);
        else if (!strcmp(argv[i], "--web-port") && i + 1 < argc) web_port    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--spots-config") && i + 1 < argc) parking_config_path = argv[++i];
        else if (!strcmp(argv[i], "--verbose"))  verbose = true;
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); print_usage(argv[0]); return 1; }
    }

    static constexpr rknn_core_mask kCoreMap[] = {
        RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2,
    };

    /* Build worker descriptors: {core_mask} for each thread to create */
    struct WorkerDesc { rknn_core_mask mask; };
    std::vector<WorkerDesc> worker_descs;

    if (core_spec < 0) {
        /* Auto: N threads, thread i pinned to NPU core i */
        int n = npu_core_count();
        for (int i = 0; i < n; ++i)
            worker_descs.push_back({kCoreMap[i]});
    } else {
        /* Single thread pinned to the specified core */
        if (core_spec > 2) {
            fprintf(stderr, "Invalid --npu value %d (valid: -1, 0, 1, 2)\n", core_spec);
            return 1;
        }
        worker_descs.push_back({kCoreMap[core_spec]});
    }

    int num_workers = (int)worker_descs.size();

    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    printf("=== RTSP YOLOv8 NPU Pipeline ===\n");
    printf("  URL      : %s\n", rtsp_url.c_str());
    printf("  Model    : %s\n", model_path.c_str());
    if (core_spec < 0)
        printf("  NPU      : %d thread(s), each pinned to core 0..%d\n", num_workers, num_workers - 1);
    else
        printf("  NPU      : 1 thread, pinned to core %d\n", core_spec);
    printf("  Transport: %s\n", transport.c_str());
    if (web_port > 0)
        printf("  Web view : http://<device-ip>:%d\n", web_port);
    if (!parking_config_path.empty())
        printf("  Parking  : %s\n", parking_config_path.c_str());
    printf("\n");

    /* ── MJPEG web server (optional) ── */
    std::unique_ptr<MjpegServer> web_server;
    if (web_port > 0) {
        web_server = std::make_unique<MjpegServer>(web_port);
        web_server->start();
    }

    BoundedQueue<AVFrame *>   frame_queue(num_workers * 3 + 4);
    BoundedQueue<InferResult> result_queue(num_workers * 4);

    Decoder decoder(frame_queue);
    DecoderConfig dec_cfg;
    dec_cfg.url            = rtsp_url;
    dec_cfg.rtsp_transport = transport;
    dec_cfg.num_hw_frames  = 16 + num_workers * 4;

    VideoInfo vinfo{};
    if (!decoder.init(dec_cfg, vinfo)) {
        fprintf(stderr, "Failed to initialise decoder\n");
        return 1;
    }
    print_video_info(vinfo);

    /* ── Parking spot config (optional) ── */
    std::vector<ParkingSpot> parking_spots;
    if (!parking_config_path.empty()) {
        parking_spots = load_parking_config(parking_config_path);
        if (parking_spots.empty()) {
            fprintf(stderr, "Error: failed to load parking config from %s\n",
                    parking_config_path.c_str());
            return 1;
        }
        printf("Parking spots: %zu loaded from %s\n",
               parking_spots.size(), parking_config_path.c_str());
    }

    std::vector<std::unique_ptr<NpuWorker>> workers;
    workers.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
        auto w = std::make_unique<NpuWorker>(i, frame_queue, result_queue);
        NpuWorkerConfig wcfg;
        wcfg.core_mask  = worker_descs[i].mask;
        wcfg.model_path = model_path;
        wcfg.model_w    = 640;
        wcfg.model_h    = 640;
        if (!w->init(wcfg)) {
            fprintf(stderr, "Failed to initialise NPU worker %d\n", i);
            return 1;
        }
        /* Only worker 0 handles web encoding to avoid redundant work */
        if (web_server && i == 0)
            w->set_mjpeg_server(web_server.get());
        if (!parking_spots.empty())
            w->set_parking_spots(parking_spots);
        workers.push_back(std::move(w));
    }

    for (auto &w : workers) w->start();
    decoder.start();

    printf("Pipeline running. Press Ctrl+C to stop.\n");
    printf("%s\n", std::string(70, '-').c_str());

    StatsTracker stats;
    InferResult  result;

    while (g_running) {
        /* Use timed pop so Ctrl+C (g_running=0) is checked every 100ms even
         * when the result queue is empty (e.g. NPU busy or pipeline starting). */
        if (!result_queue.pop_timeout(result, 100)) {
            /* timeout → re-check g_running; or queue closed → exit */
            if (result_queue.is_closed()) break;
            continue;
        }

        stats.record(result);

        /* Always print detections */
        if (!result.dets.empty()) {
            printf("  [frame %6lld]", (long long)result.frame_id);
            for (const auto &d : result.dets) {
                printf("  %s(%.2f)[%.0f,%.0f,%.0f,%.0f]",
                       d.label, d.conf,
                       d.box.left, d.box.top, d.box.right, d.box.bottom);
            }
            printf("\n");
        }

        /* Parking occupancy summary */
        if (!result.occupancy.empty()) {
            printf("  [frame %6lld]  parking:", (long long)result.frame_id);
            for (const auto &occ : result.occupancy) {
                printf("  %s=%s", occ.label,
                       occ.occupied ? "OCCUPIED" : "empty");
            }
            printf("\n");
        }

        StatsSnapshot snap{};
        if (stats.snapshot(interval_s, snap))
            print_stats(snap, num_workers, verbose, decoder.dropped_frames());
    }

    printf("\nShutting down...\n");
    fflush(stdout);

    frame_queue.close();
    result_queue.close();

    if (web_server) web_server->stop();

    decoder.stop();
    for (auto &w : workers) w->stop();

    StatsSnapshot snap{};
    stats.snapshot(0.0, snap);
    printf("\n=== Session Summary ===\n");
    printf("  Total time      : %.1f s\n",   snap.wall_s);
    printf("  Total frames    : %lld\n",     (long long)snap.total_frames);
    printf("  Dropped frames  : %lld\n",     (long long)decoder.dropped_frames());
    printf("  Avg FPS         : %.2f\n",     snap.avg_fps);
    printf("  Avg RGA         : %.2f ms\n",  snap.avg_scale_ms);
    printf("  Avg NPU infer   : %.2f ms\n",  snap.avg_infer_ms);
    printf("  Avg post-proc   : %.2f ms\n",  snap.avg_postproc_ms);
    printf("  Total detections: %lld\n",     (long long)snap.total_dets);
    printf("  CPU usage       : %.1f%%\n",   snap.cpu_pct);
    printf("  NPU workers     : %d\n",       num_workers);
    printf("=======================\n");

    return 0;
}
