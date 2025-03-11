#include <cuda_runtime.h>
#include <iostream>
#include <unistd.h>
#include "Handle/Cuda_Ipc_Manager.h"

#define META_SHM_THRESHOLD "threshold_meta"
#define DATA_SHM_THRESHOLD "threshold_data"
#define META_SHM_FINAL "final_meta"
#define DATA_SHM_FINAL "final_data"

__global__ void thresholdKernel(unsigned char* input, unsigned char* output, int width, int height, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    output[idx] = (input[idx] > threshold) ? 255 : 0;
}

void applyThreshold() {
    CudaIpcManager ipc_threshold(META_SHM_THRESHOLD, DATA_SHM_THRESHOLD);
    CudaIpcManager ipc_final(META_SHM_FINAL, DATA_SHM_FINAL);

    // Import GPU memory from threshold input
    unsigned char* d_input = (unsigned char*)ipc_threshold.importMemory(READ);
    if (!d_input) {
        std::cerr << "Error: Failed to import GPU memory!" << std::endl;
        return;
    }

    // Assume image dimensions (should be shared properly)
    int width = 675;
    int height = 225;
    size_t pitch;

    // Allocate GPU memory for output with pitch
    unsigned char* d_output;
    if (cudaMallocPitch((void**)&d_output, &pitch, width , height) != cudaSuccess) {
        std::cerr << "Error: Failed to allocate GPU memory for output!" << std::endl;
        return;
    }


    // Launch CUDA Kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    thresholdKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, 128);
    cudaDeviceSynchronize();

    // Export processed image to the final stage
    ipc_final.exportMemory(d_output, pitch * height);

    std::cout << "Thresholding (CUDA): Applied and exported to final stage." << std::endl;
    sleep(10);
    // Free GPU memory
    cudaFree(d_output);
}

int main() {
    std::cout << "Thresholding (CUDA): Processing..." << std::endl;
    applyThreshold();
    return 0;
}
