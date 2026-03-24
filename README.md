# rtsp_yolo

RTSP → RKMPP 硬件解码 → RGA 缩放 → RKNN YOLOv8 推理，面向 RK3576 / RK3588 的零拷贝实时目标检测流水线。

---

## 功能特性

- **RTSP 拉流**：支持 TCP / UDP，FFmpeg + h264_rkmpp / hevc_rkmpp 硬件解码
- **零拷贝**：RKMPP DMA buf 直接送入 RGA，RGA 输出直接写入 RKNN 输入 buffer，CPU 不参与像素搬运
- **实时性**：推理速度低于流帧率时，自动丢弃最旧帧，始终处理最新画面
- **多 NPU 核心**：可配置 1~3 个 worker 线程，兼容 RK3576（2 核）和 RK3588（3 核）
- **断流自动重连**：EOF / 网络错误 / 读超时均触发重连，指数退避，无需外部守护进程
- **MJPEG 调试查看器**：可选开启，浏览器实时查看带检测框的标注画面

---

## 数据流

```
[RTSP 网络]
    │ TCP/UDP
    ▼
[Decoder 线程]
  FFmpeg avformat + h264_rkmpp
  输出: AVFrame (AV_PIX_FMT_DRM_PRIME, DMA fd)
  断流时自动重连（指数退避）
    │
    │  BoundedQueue<AVFrame*>
    │  push_latest()：队列满时淘汰最旧帧保持实时
    ▼
[NpuWorker 线程 × N]
  ├─ RGA：DMA fd → 640×640 RGB888（零拷贝，写入 RKNN 输入 DMA buffer）
  ├─ rknn_run()：INT8 推理，RKNN 内部自动完成量化
  ├─ YOLOv8 后处理：双遍缓存友好扫描 + NMS
  └─ （可选）MJPEG 编码 → MjpegServer
    │
    │  BoundedQueue<InferResult>
    ▼
[Main 线程]
  打印检测结果，定期输出性能统计
```

---

## 环境要求

| 项目 | 说明 |
|------|------|
| 宿主机 | Linux x86_64 |
| 交叉编译器 | `aarch64-linux-gnu-gcc` / `g++` |
| Sysroot | `/home/miles/workspace/rockchip/sysroot_3576` |
| FFmpeg | 静态库，含 RKMPP 支持，已安装在 sysroot |
| RKNN Toolkit2 | `/home/miles/workspace/rockchip/rknn-toolkit2/` |
| 目标平台 | RK3576 / RK3588，Linux aarch64 |

---

## 构建

```bash
./build.sh
```

产物：
- `build/rtsp_yolo` — 可执行文件
- `compile_commands.json` — clangd 索引（自动软链到项目根目录）

---

## 部署

```bash
# 拷贝可执行文件和模型
scp build/rtsp_yolo <model.rknn> user@device:~/

# 拷贝 RKNN 运行时
scp /home/miles/workspace/rockchip/sysroot_3576/usr/lib/librknnrt.so user@device:~/lib/
```

---

## 运行

```bash
# 基本用法
./rtsp_yolo rtsp://192.168.1.100:8554/stream yolov8n.rknn

# 详细性能统计（每周期打印 RGA / NPU / 后处理耗时）
./rtsp_yolo <url> <model.rknn> --verbose

# 开启 MJPEG 调试查看器，浏览器访问 http://<device-ip>:8080
./rtsp_yolo <url> <model.rknn> --web-port 8080

# 指定 2 个 NPU worker（RK3576 双核 / RK3588 可用 3）
./rtsp_yolo <url> <model.rknn> --workers 2
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--transport tcp\|udp` | `tcp` | RTSP 传输协议 |
| `--workers N` | `1` | NPU worker 线程数（1~3） |
| `--interval S` | `3` | 统计打印间隔（秒） |
| `--verbose` | 关闭 | 打印每周期 RGA / NPU / 后处理耗时 |
| `--web-port N` | 关闭 | 启用 MJPEG 调试查看器 |

---

## 输出示例

```
=== RTSP YOLOv8 NPU Pipeline ===
  URL      : rtsp://192.168.1.100:8554/stream
  Model    : yolov8n.rknn
  Workers  : 1 NPU thread(s)

  [frame   120]  car(0.89)[660,213,901,362]  truck(0.45)[844,106,959,285]
  [frame   121]  car(0.82)[655,210,895,358]

[  3.0 s]  fps(inst/avg)= 19.3/ 19.3  cpu= 36.1%  dets/frame=4.6  drop=0

=== Session Summary ===
  Total time      : 15.5 s
  Total frames    : 245
  Dropped frames  : 0
  Avg FPS         : 19.43
  Avg RGA         : 1.86 ms
  Avg NPU infer   : 16.68 ms
  Avg post-proc   : 4.42 ms
  Total detections: 1172
  CPU usage       : 36.1%
  NPU workers     : 1
=======================
```

---

## 性能参考（RK3576，960×540，H.264，~2600 kbps）

| 指标 | 数值 |
|------|------|
| 解码方式 | RKMPP 硬件解码 |
| RGA 缩放耗时 | ~1.9 ms |
| NPU 推理耗时 | ~16.7 ms（1 worker，双核并行） |
| 后处理耗时 | ~4.4 ms |
| 推理帧率 | ~19 fps |
| CPU 占用 | ~36% |

---

## 项目结构

```
src/
├── main.cc            # 入口，参数解析，shutdown 逻辑
├── common.hpp         # BoundedQueue<T>，InferResult，Detection
├── decoder.cc/hpp     # RTSP 拉流 + RKMPP 硬件解码线程（含断流重连）
├── npu_worker.cc/hpp  # RGA 零拷贝缩放 + RKNN 推理 + MJPEG 编码
├── postprocess.cc/hpp # YOLOv8 后处理，双遍缓存友好扫描
├── stats.cc/hpp       # 帧率 / 延迟 / CPU 统计
└── mjpeg_server.cc/hpp # HTTP MJPEG 调试服务

cmake/
└── toolchain.cmake    # aarch64 交叉编译工具链定义

build.sh               # 构建脚本
DEVLOG.md              # 开发记录（变更历史、调试过程、关键决策）
.clangd                # clangd 跨编译头文件路径（解决 'atomic' not found）
```

---

> 开发记录、变更历史、调试过程详见 [DEVLOG.md](DEVLOG.md)


---

## 数据流

```
[RTSP 网络]
    │ TCP/UDP
    ▼
[Decoder 线程]
  FFmpeg avformat + h264_rkmpp
  输出: AVFrame (AV_PIX_FMT_DRM_PRIME, DMA fd)
  断流时自动重连（指数退避）
    │
    │  BoundedQueue<AVFrame*>
    │  push_latest()：队列满时淘汰最旧帧保持实时
    ▼
[NpuWorker 线程 × N]
  ├─ RGA：DMA fd → 640×640 RGB888（零拷贝，写入 RKNN 输入 DMA buffer）
  ├─ rknn_run()：INT8 推理，RKNN 内部自动完成量化
  ├─ YOLOv8 后处理：双遍缓存友好扫描 + NMS
  └─ （可选）MJPEG 编码 → MjpegServer
    │
    │  BoundedQueue<InferResult>
    ▼
[Main 线程]
  打印检测结果，定期输出性能统计
```

---

## 环境要求

| 项目 | 说明 |
|------|------|
| 宿主机 | Linux x86_64 |
| 交叉编译器 | `aarch64-linux-gnu-gcc` / `g++` |
| Sysroot | `/home/miles/workspace/rockchip/sysroot_3576` |
| FFmpeg | 静态库，含 RKMPP 支持，已安装在 sysroot |
| RKNN Toolkit2 | `/home/miles/workspace/rockchip/rknn-toolkit2/` |
| 目标平台 | RK3576 / RK3588，Linux aarch64 |

---

## 构建

```bash
./build.sh
```

产物：
- `build/rtsp_yolo` — 可执行文件
- `compile_commands.json` — clangd 索引（自动软链到项目根目录）

---

## 部署

```bash
# 拷贝可执行文件和模型
scp build/rtsp_yolo <model.rknn> user@device:~/

# 拷贝 RKNN 运行时
scp /home/miles/workspace/rockchip/sysroot_3576/usr/lib/librknnrt.so user@device:~/lib/
```

---

## 运行

```bash
# 基本用法
./rtsp_yolo rtsp://192.168.1.100:8554/stream yolov8n.rknn

# 详细性能统计（每周期打印 RGA / NPU / 后处理耗时）
./rtsp_yolo <url> <model.rknn> --verbose

# 开启 MJPEG 调试查看器，浏览器访问 http://<device-ip>:8080
./rtsp_yolo <url> <model.rknn> --web-port 8080

# 指定 2 个 NPU worker（RK3576 双核 / RK3588 可用 3）
./rtsp_yolo <url> <model.rknn> --workers 2
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--transport tcp\|udp` | `tcp` | RTSP 传输协议 |
| `--workers N` | `1` | NPU worker 线程数（1~3） |
| `--interval S` | `3` | 统计打印间隔（秒） |
| `--verbose` | 关闭 | 打印每周期 RGA / NPU / 后处理耗时 |
| `--web-port N` | 关闭 | 启用 MJPEG 调试查看器 |

---

## 断流重连机制

Decoder 线程内置断流自动恢复，无需外部守护进程。

### 触发条件

| 场景 | FFmpeg 返回值 | 处理 |
|------|-------------|------|
| 推流端主动停止 | `AVERROR_EOF` | 立即重连 |
| 网络断开 / 复位 | `AVERROR(ECONNRESET)` 等 | 立即重连 |
| 流量中断（TCP 无数据） | `interrupt_cb` 超时 → `AVERROR_EXIT` | 重连 |

### 重连策略

- **指数退避**：第 N 次失败后等待 `reconnect_delay_ms × 2^min(N,5)` ms，上限 30 s
- **无限重连**（默认）：`reconnect_retries = -1`；设为 `0` 可禁用重连，设正整数限制次数
- **读超时看门狗**：默认 10 s 无数据触发；`read_timeout_s = 0` 可关闭
- **RKMPP 上下文复用**：`hw_dev_` 创建一次，重连时只重建 format / codec context

### 日志示例

```
[Decoder] Stream EOF (sender disconnected); will reconnect
[Decoder] Reconnecting in 1000 ms (attempt 1, unlimited)...
[Decoder] Opening stream: rtsp://192.168.1.100:8554/stream
[Decoder] Stream ready: 1920x1080  25.00 fps  codec=h264_rkmpp  hw=RKMPP
[Decoder] Reconnected successfully (was attempt 1)
```

### DecoderConfig 参数

```cpp
struct DecoderConfig {
    std::string rtsp_transport     = "tcp";
    int         num_hw_frames      = 16;
    int         reconnect_retries  = -1;    // -1=无限, 0=禁用, N=最多N次
    int         reconnect_delay_ms = 1000;  // 基础退避时间（ms）
    int         read_timeout_s     = 10;    // 读超时看门狗（s），0=关闭
};
```

---

## 输出示例

```
=== RTSP YOLOv8 NPU Pipeline ===
  URL      : rtsp://192.168.1.100:8554/stream
  Model    : yolov8n.rknn
  Workers  : 1 NPU thread(s)

  [frame   120]  car(0.89)[660,213,901,362]  truck(0.45)[844,106,959,285]
  [frame   121]  car(0.82)[655,210,895,358]

[  3.0 s]  fps(inst/avg)= 19.3/ 19.3  cpu= 36.1%  dets/frame=4.6  drop=0

=== Session Summary ===
  Total time      : 15.5 s
  Total frames    : 245
  Dropped frames  : 0
  Avg FPS         : 19.43
  Avg RGA         : 1.86 ms
  Avg NPU infer   : 16.68 ms
  Avg post-proc   : 4.42 ms
  Total detections: 1172
  CPU usage       : 36.1%
  NPU workers     : 1
=======================
```

---

## 性能参考（RK3576，960×540，H.264，~2600 kbps）

| 指标 | 数值 |
|------|------|
| 解码方式 | RKMPP 硬件解码 |
| RGA 缩放耗时 | ~1.9 ms |
| NPU 推理耗时 | ~16.7 ms（1 worker，双核并行） |
| 后处理耗时 | ~4.4 ms |
| 推理帧率 | ~19 fps |
| CPU 占用 | ~36% |

---

## 项目结构

```
src/
├── main.cc            # 入口，参数解析，shutdown 逻辑
├── common.hpp         # BoundedQueue<T>，InferResult，Detection
├── decoder.cc/hpp     # RTSP 拉流 + RKMPP 硬件解码线程
├── npu_worker.cc/hpp  # RGA 零拷贝缩放 + RKNN 推理 + MJPEG 编码
├── postprocess.cc/hpp # YOLOv8 后处理，双遍缓存友好扫描
├── stats.cc/hpp       # 帧率 / 延迟 / CPU 统计
└── mjpeg_server.cc/hpp # HTTP MJPEG 调试服务

cmake/
└── toolchain.cmake    # aarch64 交叉编译工具链定义

build.sh               # 构建脚本
.clangd                # clangd 跨编译头文件路径（解决 'atomic' not found）
```
