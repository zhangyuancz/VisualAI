cmake_minimum_required(VERSION 3.16)
project(rtsp_yolo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clangd

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# ── Sysroot paths ─────────────────────────────────────────────────────────────
set(SYSROOT /home/miles/workspace/rockchip/sysroot_3576)
set(RKNN_RT_DIR /home/miles/workspace/rockchip/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api)

# ── Compiler flags ────────────────────────────────────────────────────────────
add_compile_options(-march=armv8-a -Wall -Wextra -Wno-unused-parameter)
include_directories(SYSTEM
    ${SYSROOT}/usr/include
    ${SYSROOT}/usr/include/libdrm
    ${SYSROOT}/usr/include/rockchip
    ${SYSROOT}/usr/include/rga
    ${RKNN_RT_DIR}/include
)

link_directories(
    ${SYSROOT}/usr/lib
    ${SYSROOT}/usr/lib/aarch64-linux-gnu
)

# ── RKNN Runtime ──────────────────────────────────────────────────────────────
set(RKNN_RT_LIB ${SYSROOT}/usr/lib/librknnrt.so)

# ── Sources ───────────────────────────────────────────────────────────────────
add_executable(rtsp_yolo
    src/main.cc
    src/decoder.cc
    src/npu_worker.cc
    src/postprocess.cc
    src/stats.cc
    src/mjpeg_server.cc
)

target_include_directories(rtsp_yolo PRIVATE src)

# ── Link libraries ────────────────────────────────────────────────────────────
target_link_libraries(rtsp_yolo
    # FFmpeg (static)
    -lavformat -lavcodec -lavutil -lswresample -lswscale
    # Rockchip hardware
    -lrockchip_mpp
    -lrga
    -ldrm
    ${RKNN_RT_LIB}
    # System
    -pthread -lm -latomic
)

# Embed RPATH so the binary finds librknnrt.so in ./lib at runtime
set_target_properties(rtsp_yolo PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN/lib"
    LINK_FLAGS "-Wl,--allow-shlib-undefined"
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS rtsp_yolo DESTINATION .)
install(FILES ${RKNN_RT_LIB} DESTINATION lib)
