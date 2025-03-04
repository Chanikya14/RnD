#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "cuda_ipc_manager.h"

extern "C" void displayFrame(CudaIpcManager &ipcManager, int width, int height) {
    void *d_frame = ipcManager.importMemory(READ);
    if (!d_frame) return;

    cv::Mat frame(height, width, CV_8UC1);
    cudaMemcpy(frame.data, d_frame, width * height, cudaMemcpyDeviceToHost);
    
    cv::imshow("Processed Video", frame);
    cv::waitKey(1);
}
