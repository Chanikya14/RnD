#include <cuda_runtime.h>
#include "cuda_ipc_manager.h"

__global__ void grayscaleKernel(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        output[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

extern "C" void preprocessFrame(CudaIpcManager &ipcManager, int width, int height) {
    void *d_frame = ipcManager.importMemory(WRITE);
    if (!d_frame) return;

    unsigned char *d_gray;
    cudaMalloc(&d_gray, width * height);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    grayscaleKernel<<<gridSize, blockSize>>>((unsigned char*)d_frame, d_gray, width, height, 3);

    cudaMemcpy(d_frame, d_gray, width * height, cudaMemcpyDeviceToDevice);
    cudaFree(d_gray);
}
