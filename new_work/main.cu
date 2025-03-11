#include <opencv2/opencv.hpp>
#include <iostream>
#include <signal.h>
#include <unistd.h>
#include "Handle/Cuda_Ipc_Manager.h"

#define META_SHM "median_meta"
#define DATA_SHM "median_data"

void startPipeline() {
    // Read image from disk
    cv::Mat image = cv::imread("nebula.jpeg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Failed to read nebula.jpeg!" << std::endl;
        return;
    }

    // Upload image to GPU
    cv::cuda::GpuMat d_image;
    d_image.upload(image);

    // Export the GPU memory handle using CudaIpcManager
    CudaIpcManager ipc(META_SHM, DATA_SHM);
    ipc.exportMemory(d_image.data, d_image.step * d_image.rows);

    std::cout << "Main: GPU memory allocated and exported. Signaling median filter..." << std::endl;

    // Signal the median filter process
    // kill(std::stoi(std::string(std::getenv("MEDIAN_PID"))), SIGUSR1);
}

int main() {
    startPipeline();
    return 0;
}
