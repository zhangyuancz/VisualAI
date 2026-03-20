/**
 * mjpeg_server.hpp — Minimal MJPEG-over-HTTP server.
 *
 * Any browser can view the stream by opening:
 *   http://<device_ip>:<port>
 *
 * Thread-safe: push_frame() can be called from any thread.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

class MjpegServer {
public:
    explicit MjpegServer(int port);
    ~MjpegServer();

    void start();
    void stop();

    /* Push a new JPEG frame to all connected clients. Thread-safe. */
    void push_frame(std::vector<uint8_t> jpeg);

    int port() const { return port_; }

private:
    void accept_loop();
    void client_loop(int client_fd);

    int  port_;
    int  server_fd_ = -1;

    std::atomic<bool> stop_req_{false};
    std::thread       accept_thread_;

    /* Shared frame state */
    std::mutex              mtx_;
    std::condition_variable cv_;
    std::vector<uint8_t>    frame_;
    uint64_t                seq_ = 0;
};
