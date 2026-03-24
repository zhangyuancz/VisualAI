# 开发记录

本文档记录每次提交的变更内容、关键设计决策及调试过程。

---

## [cb671cd] 初始提交：零拷贝推理流水线

### 目标

在 RK3576 / RK3588 上实现 RTSP → NPU 的端到端实时目标检测，全程不经过 CPU 搬运像素。

### 架构设计

**零拷贝链路：**
```
RKMPP 解码 → DMA fd (AVDRMFrameDescriptor)
    └─ RGA importbuffer_fd() 导入源 buffer
    └─ rknn_create_mem() 分配目标 buffer（同时是 RKNN 输入）
    └─ improcess() 缩放直接写入 RKNN 输入 DMA
    └─ rknn_run() 直接读取，无额外拷贝
```

**BoundedQueue 实时语义：**
- `push_latest()`：队列满时淘汰最旧帧而非阻塞，保证推理始终处理最新画面
- `pop_timeout()`：主线程带超时轮询，响应 Ctrl+C 不阻塞

**多 Worker 核心绑定：**
- 1 个 worker：`RKNN_NPU_CORE_ALL`，由驱动分配
- 多个 worker：按 `id % 3` 绑定到独立核心，避免竞争

**MJPEG 调试服务：**
- Worker 0 负责 web 编码，避免多 worker 重复编码
- 采用 `mmap` 读取 DMA buffer 后立即 `munmap`，不持有 fd

### 关键参数选择

| 参数 | 值 | 原因 |
|------|----|------|
| `extra_hw_frames` | `16 + workers*4` | 防止 RKMPP 帧池耗尽导致解码阻塞 |
| `stimeout` | 5,000,000 µs | RTSP 连接超时 5s，平衡响应速度与网络抖动容忍 |
| `AV_CODEC_FLAG_LOW_DELAY` | 开启 | 降低解码延迟，适合实时场景 |
| MJPEG quality | `FF_QP2LAMBDA * 7` | 约 85% 质量，调试可读且带宽可控 |

---

## [当前工作分支] RTSP 断流鲁棒性

### 问题背景

推流端重启后程序无任何响应，需要手动重启。分析发现存在多个叠加问题。

### 问题分析

#### 问题一（核心）：`AVERROR_EOF` 未触发重连

```cpp
// 原代码
if (ret == AVERROR_EOF) break;          // 直接退出内层循环
...
if (!need_reconnect) break;             // 当作正常结束，退出外层循环
```

推流端主动停止时，FFmpeg 返回 `AVERROR_EOF`。原代码将其视为正常流结束，解码线程退出，pipeline 永久停止。这是程序"无响应"的根本原因。

#### 问题二：`last_pkt_us_` 过期导致重连期间看门狗误触发

看门狗（`interrupt_cb`）检测距上次收包时间是否超过 `read_timeout_s`。流断开后，`last_pkt_us_` 停留在最后一帧的时间戳。

重连流程：
1. 超时触发（10s 后）
2. 进入重连，等待退避时间（1s）
3. 调用 `close_stream()` + `open_stream()`
4. `open_stream()` 内部调用 `avformat_find_stream_info()`
5. `find_stream_info` 调用 `interrupt_cb` → elapsed ≈ 11s > 10s → **返回 1，中断**
6. `find_stream_info` 失败 → 重连失败 → 无限循环但永远连不上

修复：在 `open_stream()` 入口处立即重置 `last_pkt_us_.store(now_us())`，连接建立后再刷新一次。

#### 问题三：`avformat_open_input` 无法被 `stop_req_` 打断

原代码在 `avformat_open_input` 返回后才挂载 `interrupt_cb`，导致 `stop()` 被调用时，正在阻塞的连接尝试无法中断（最长等待 `stimeout` = 5s）。

修复：预分配 `fmt_ctx_`，在调用 `avformat_open_input` **之前**挂载回调：
```cpp
fmt_ctx_ = avformat_alloc_context();
fmt_ctx_->interrupt_callback.callback = Decoder::interrupt_cb;
fmt_ctx_->interrupt_callback.opaque   = this;
ret = avformat_open_input(&fmt_ctx_, url, nullptr, &opts);
```

#### 问题四：重连失败后 `fmt_ctx_` 为 nullptr 导致 segfault

`avformat_open_input` 失败时，FFmpeg 内部调用 `avformat_free_context(s)` 并将 `*ps` 设为 `NULL`。`open_stream()` 返回 false 后，`run()` 执行 `continue` 回到外层循环，下一轮直接调用 `av_read_frame(nullptr, pkt)` → **Segmentation fault**。

实际崩溃日志：
```
[Decoder] avformat_open_input failed: Server returned 404 Not Found
[Decoder] Reconnect attempt 1 failed; will retry
Segmentation fault
```

修复：在内层读循环入口加 nullptr 守卫：
```cpp
while (!stop_req_.load() && !queue_closed) {
    if (!fmt_ctx_) { need_reconnect = true; break; }  // 新增
    int ret = av_read_frame(fmt_ctx_, pkt);
    ...
}
```
`fmt_ctx_` 为空时立即设 `need_reconnect = true`，走指数退避后再次尝试，不进入读逻辑。

### 最终设计

**重连状态机（`Decoder::run()` 外层循环）：**

```
┌─────────────────────────────────────────┐
│  外层循环 while (!stop_req_)             │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  if (!fmt_ctx_) → need_reconnect │   │◄─┐
│  │  内层读循环                       │   │  │
│  │    av_read_frame()               │   │  │ open 失败
│  │    EOF/error → need_reconnect    │   │  │ retries++
│  └──────────────────────────────────┘   │  │
│                                         │  │
│  if queue_closed / stop_req_ → exit     │  │
│                                         │  │
│  ── 指数退避 sleep ──                    │  │
│  close_stream()                         │  │
│  open_stream() ─── 成功 → retries=0 ───►│  │
│               └── 失败 ─────────────────┘  │
└─────────────────────────────────────────┘
```

**`DecoderConfig` 新增参数：**

```cpp
int reconnect_retries  = -1;    // -1=无限, 0=禁用, N=最多N次
int reconnect_delay_ms = 1000;  // 基础退避（ms），实际 = delay * 2^min(retries,5)，上限 30s
int read_timeout_s     = 10;    // 读超时看门狗，0=关闭
```

**RKMPP 上下文复用：** `hw_dev_`（`av_hwdevice_ctx`）仅在 `init()` 创建一次，`close_stream()` / `open_stream()` 只操作 `fmt_ctx_` 和 `dec_ctx_`，避免重连时重新初始化 RKMPP 驱动。

### 断流重连日志示例

```
[Decoder] Decode thread started (reconnect=on max=-1 timeout=10s)
[Decoder] Stream ready: 1920x1080  25.00 fps  codec=h264_rkmpp  hw=RKMPP

# 推流端停止
[Decoder] Stream EOF (sender disconnected); will reconnect
[Decoder] Reconnecting in 1000 ms (attempt 1, unlimited)...
[Decoder] Opening stream: rtsp://172.16.41.214:8554/mystream

# 推流端尚未恢复，重连失败
[Decoder] avformat_open_input failed: Server returned 404 Not Found
[Decoder] Reconnect attempt 1 failed; will retry
[Decoder] Reconnecting in 2000 ms (attempt 2, unlimited)...
[Decoder] Opening stream: rtsp://172.16.41.214:8554/mystream

# 推流端恢复
[Decoder] Stream ready: 1920x1080  25.00 fps  codec=h264_rkmpp  hw=RKMPP
[Decoder] Reconnected successfully (was attempt 2)
```
