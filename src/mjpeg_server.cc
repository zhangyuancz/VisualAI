/**
 * mjpeg_server.cc — Minimal MJPEG-over-HTTP server implementation.
 *
 * Protocol: HTTP multipart/x-mixed-replace.
 * Each client connection is served in a detached thread.
 */
#include "mjpeg_server.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

static const char kHttpHeader[] =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: multipart/x-mixed-replace; boundary=mjpeg\r\n"
    "Cache-Control: no-cache, no-store\r\n"
    "Connection: close\r\n"
    "\r\n";

/* Simple HTML landing page — browser that hits "/" gets this */
static const char kHtmlPage[] =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/html\r\n"
    "Connection: close\r\n"
    "\r\n"
    "<!DOCTYPE html><html><head><title>RTSP YOLOv8 Debug</title>"
    "<style>body{margin:0;background:#000;display:flex;justify-content:center;"
    "align-items:center;height:100vh;}"
    "img{max-width:100%;max-height:100vh;}</style></head>"
    "<body><img src='/stream' /></body></html>\r\n";

MjpegServer::MjpegServer(int port) : port_(port) {}

MjpegServer::~MjpegServer() { stop(); }

void MjpegServer::start()
{
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) { perror("[MjpegServer] socket"); return; }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(port_));

    if (bind(server_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        perror("[MjpegServer] bind");
        close(server_fd_); server_fd_ = -1;
        return;
    }
    listen(server_fd_, 8);

    stop_req_.store(false);
    accept_thread_ = std::thread(&MjpegServer::accept_loop, this);
    fprintf(stderr, "[MjpegServer] Listening on http://0.0.0.0:%d\n", port_);
}

void MjpegServer::stop()
{
    stop_req_.store(true);
    cv_.notify_all();
    if (server_fd_ >= 0) {
        /* shutdown() reliably unblocks accept() in another thread on Linux;
         * close() alone is not guaranteed to do so. */
        shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
    }
    if (accept_thread_.joinable()) accept_thread_.join();
}

void MjpegServer::push_frame(std::vector<uint8_t> jpeg)
{
    {
        std::lock_guard<std::mutex> lk(mtx_);
        frame_ = std::move(jpeg);
        ++seq_;
    }
    cv_.notify_all();
}

void MjpegServer::accept_loop()
{
    while (!stop_req_.load()) {
        sockaddr_in caddr{};
        socklen_t   clen = sizeof(caddr);
        int cfd = accept(server_fd_, reinterpret_cast<sockaddr *>(&caddr), &clen);
        if (cfd < 0) break;
        std::thread([this, cfd] { client_loop(cfd); }).detach();
    }
}

void MjpegServer::client_loop(int fd)
{
    /* Read HTTP request line to determine path */
    char req[512] = {};
    recv(fd, req, sizeof(req) - 1, 0);

    /* Serve HTML page for "/" or "/index.html" */
    if (strstr(req, "GET /stream") == nullptr) {
        send(fd, kHtmlPage, sizeof(kHtmlPage) - 1, MSG_NOSIGNAL);
        close(fd);
        return;
    }

    /* Stream MJPEG for "/stream" */
    send(fd, kHttpHeader, sizeof(kHttpHeader) - 1, MSG_NOSIGNAL);

    uint64_t last_seq = 0;
    while (!stop_req_.load()) {
        std::vector<uint8_t> jpeg;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [&] { return seq_ != last_seq || stop_req_.load(); });
            if (stop_req_.load()) break;
            jpeg     = frame_;
            last_seq = seq_;
        }

        char part_hdr[128];
        int  hlen = snprintf(part_hdr, sizeof(part_hdr),
                             "--mjpeg\r\n"
                             "Content-Type: image/jpeg\r\n"
                             "Content-Length: %zu\r\n\r\n",
                             jpeg.size());

        if (send(fd, part_hdr, hlen, MSG_NOSIGNAL) < 0) break;
        if (send(fd, jpeg.data(), jpeg.size(), MSG_NOSIGNAL) < 0) break;
        send(fd, "\r\n", 2, MSG_NOSIGNAL);
    }
    close(fd);
}
