#include <opencv2/opencv.hpp>
#include <iostream>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include "Handle/Cuda_Ipc_Manager.h"
#include <cuda_runtime.h>

#define META_SHM "median_meta"
#define DATA_SHM "median_data"

void startPipeline() {
    // Read image from disk
    cv::Mat image = cv::imread("nebula.jpeg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Failed to read nebula.jpeg!" << std::endl;
        return;
    }

    uchar* d_image;  // Pointer for GPU memory
    size_t pitch;
    size_t width = image.cols * image.elemSize();  // Total width in bytes
    size_t height = image.rows;

    std::cout <<width<<" "<<height<<std::endl;
    // Allocate GPU memory with proper alignment
    cudaMallocPitch(&d_image, &pitch, width, height);

    // Copy data from CPU to GPU
    cudaMemcpy2D(d_image, pitch, image.data, width, width, height, cudaMemcpyHostToDevice);

    // Export the GPU memory handle using CudaIpcManager
    CudaIpcManager ipc(META_SHM, DATA_SHM);
    ipc.exportMemory(d_image, pitch * height);

    std::cout << "Main: GPU memory allocated and exported. Signaling median filter..." << std::endl;

    sleep(5);
    // Free GPU memory (if no further processing is needed in this process)
    cudaFree(d_image);
}

int main() {
    startPipeline();
    return 0;
}
