#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <unistd.h>
#include "Handle/Cuda_Ipc_Manager.h"

#define META_SHM_MEDIAN "median_meta"
#define DATA_SHM_MEDIAN "median_data"
#define META_SHM_SHARPEN "sharpen_meta"
#define DATA_SHM_SHARPEN "sharpen_data"

#define KERNEL_SIZE 5  // 5x5 median filter

__device__ unsigned char getMedian(unsigned char* window, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (window[j] > window[j + 1]) {
                unsigned char temp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = temp;
            }
        }
    }
    return window[size / 2];
}

__global__ void medianFilterKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_k = KERNEL_SIZE / 2;
    int idx = (y * width + x) * channels;

    for (int c = 0; c < channels; c++) {
        unsigned char window[KERNEL_SIZE * KERNEL_SIZE];
        int count = 0;

        for (int dy = -half_k; dy <= half_k; dy++) {
            for (int dx = -half_k; dx <= half_k; dx++) {
                int nx = min(max(x + dx, 0), width - 1);
                int ny = min(max(y + dy, 0), height - 1);
                window[count++] = input[(ny * width + nx) * channels + c];
            }
        }
        output[idx + c] = getMedian(window, KERNEL_SIZE * KERNEL_SIZE);
    }
}

void applyMedianFilter() {
    CudaIpcManager ipc_median(META_SHM_MEDIAN, DATA_SHM_MEDIAN);
    CudaIpcManager ipc_sharpen(META_SHM_SHARPEN, DATA_SHM_SHARPEN);

    // Import GPU memory from median input
    unsigned char* d_input = (unsigned char*)ipc_median.importMemory(READ);
    if (!d_input) {
        std::cerr << "Error: Failed to import GPU memory!" << std::endl;
        return;
    }

    // TODO: Assume image dimensions (should ideally be shared metadata)
    int width = 675;  // Example values
    int height = 225;
    int channels = 3;
    size_t pitch; // Pitch value

    // Allocate GPU memory for output with pitch
    unsigned char* d_output;
    if (cudaMallocPitch((void**)&d_output, &pitch, width, height) != cudaSuccess) {
        std::cerr << "Error: Failed to allocate GPU memory for output!" << std::endl;
        return;
    }

    // Launch CUDA Kernel (updated for pitch)
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    medianFilterKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    cudaDeviceSynchronize();

    // Debugging
    std::cout << "Here" << std::endl;
    if (!d_output) std::cerr << "NULL pointer detected!" << std::endl;

    // Export processed image to the next stage (sharpen filter)
    ipc_sharpen.exportMemory(d_output, pitch * height);

    std::cout << "Median Filter (CUDA): Applied and exported to sharpen filter." << std::endl;
    sleep(10);

    // Free GPU memory
    cudaFree(d_output);
}


int main() {
    std::cout << "Median Filter (CUDA): Processing..." << std::endl;
    applyMedianFilter();
    return 0;
}
