#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "cuda_ipc_manager.h"

extern "C" void captureFrame(cv::VideoCapture &cap, CudaIpcManager &ipcManager, size_t size) {
    cv::Mat frame;
    cap.read(frame);
    if (frame.empty()) return;

    // Allocate GPU memory and copy frame
    void *d_frame;
    cudaMalloc(&d_frame, size);
    cudaMemcpy(d_frame, frame.data, size, cudaMemcpyHostToDevice);

    // Export memory handle
    ipcManager.exportMemory(d_frame, size);
}
