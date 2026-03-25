# rtsp_yolo

RTSP → RKMPP 硬件解码 → RGA 缩放 → RKNN YOLOv8 推理，面向 RK3576 / RK3588 的零拷贝实时目标检测流水线。

---

## 功能特性

- **RTSP 拉流**：支持 TCP / UDP，FFmpeg + h264_rkmpp / hevc_rkmpp 硬件解码
- **零拷贝**：RKMPP DMA buf 直接送入 RGA，RGA 输出直接写入 RKNN 输入 buffer，CPU 不参与像素搬运
- **实时性**：推理速度低于流帧率时，自动丢弃最旧帧，始终处理最新画面
- **多 NPU 核心**：`--npu -1` 自动按 SoC 核心数（RK3576=2，RK3588=3）创建对应线程，全部使用 `RKNN_NPU_CORE_AUTO` 由驱动动态调度
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

# 自动按 SoC 核心数分配 NPU 线程（默认行为，-1 = auto）
./rtsp_yolo <url> <model.rknn> --npu -1

# 固定使用 NPU core 0（单线程调试）
./rtsp_yolo <url> <model.rknn> --npu 0
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--transport tcp\|udp` | `tcp` | RTSP 传输协议 |
| `--npu -1\|0\|1\|2` | `-1` | NPU 核心选择：-1=自动适配 SoC，0/1/2=固定单核 |
| `--interval S` | `3` | 统计打印间隔（秒） |
| `--verbose` | 关闭 | 打印每周期 RGA / NPU / 后处理耗时 |
| `--web-port N` | 关闭 | 启用 MJPEG 调试查看器 |

---

## 输出示例

```
=== RTSP YOLOv8 NPU Pipeline ===
  URL      : rtsp://192.168.1.100:8554/stream
  Model    : yolov8n.rknn
  Workers  : 3 thread(s), RKNN_NPU_CORE_AUTO (SoC-adaptive)

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
  NPU workers     : 3 (RKNN_NPU_CORE_AUTO)
=======================
```

---

## 性能参考（RK3576，960×540，H.264，~2600 kbps）

| 指标 | 数值 |
|------|------|
| 解码方式 | RKMPP 硬件解码 |
| RGA 缩放耗时 | ~1.9 ms |
| NPU 推理耗时 | ~16.7 ms（--npu -1，双核 AUTO 调度）|
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

