#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TOOLCHAIN="${SCRIPT_DIR}/cmake/toolchain.cmake"

cmake -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -G "Unix Makefiles" \
    "${SCRIPT_DIR}"

cmake --build "${BUILD_DIR}" -- -j"$(nproc)"

# Symlink compile_commands.json to project root for clangd
ln -sf "${BUILD_DIR}/compile_commands.json" "${SCRIPT_DIR}/compile_commands.json"

echo ""
echo "Build complete: ${BUILD_DIR}/rtsp_yolo"
echo "  clangd index: ${SCRIPT_DIR}/compile_commands.json"
echo ""
echo "Deploy to device:"
echo "  scp ${BUILD_DIR}/rtsp_yolo <model.rknn> user@device:~/"
echo "  scp /home/miles/workspace/rockchip/sysroot_3576/usr/lib/librknnrt.so user@device:~/lib/"
echo ""
echo "Run:"
echo "  ./rtsp_yolo rtsp://<ip>:8554/stream <model.rknn> --workers 2"
