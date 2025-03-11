#include <cuda_runtime.h>
#include <iostream>
#include <unistd.h>
#include "Handle/Cuda_Ipc_Manager.h"

#define META_SHM_SHARPEN "sharpen_meta"
#define DATA_SHM_SHARPEN "sharpen_data"
#define META_SHM_THRESHOLD "threshold_meta"
#define DATA_SHM_THRESHOLD "threshold_data"

__global__ void sharpenKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    int kernel[3][3] = {
        { 0, -1,  0 },
        {-1,  5, -1 },
        { 0, -1,  0 }
    };

    for (int c = 0; c < channels; c++) {
        int sum = 0;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);
                int nidx = (ny * width + nx) * channels + c;
                sum += input[nidx] * kernel[ky + 1][kx + 1];
            }
        }
        output[idx + c] = min(max(sum, 0), 255);
    }
}

void applySharpenFilter() {
    CudaIpcManager ipc_sharpen(META_SHM_SHARPEN, DATA_SHM_SHARPEN);
    CudaIpcManager ipc_threshold(META_SHM_THRESHOLD, DATA_SHM_THRESHOLD);

    // Import GPU memory from sharpen input
    unsigned char* d_input = (unsigned char*)ipc_sharpen.importMemory(READ);
    if (!d_input) {
        std::cerr << "Error: Failed to import GPU memory!" << std::endl;
        return;
    }

    // Set image dimensions
    int width = 675;
    int height = 225;
    int channels = 3;
    size_t pitch;

    // Allocate GPU memory for output with pitch
    unsigned char* d_output;
    if (cudaMallocPitch((void**)&d_output, &pitch, width , height) != cudaSuccess) {
        std::cerr << "Error: Failed to allocate GPU memory for output!" << std::endl;
        return;
    }

    // cudaMemset2D(d_output, pitch, 0, width * channels, height);

    // Launch CUDA Kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    sharpenKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    cudaDeviceSynchronize();

    // Export processed image to the next stage (threshold filter)
    ipc_threshold.exportMemory(d_output, pitch * height);

    std::cout << "Sharpen Filter (CUDA): Applied and exported to threshold filter." << std::endl;
    sleep(10);
    // Free GPU memory
    cudaFree(d_output);
}

int main() {
    std::cout << "Sharpen Filter (CUDA): Processing..." << std::endl;
    applySharpenFilter();
    return 0;
}
