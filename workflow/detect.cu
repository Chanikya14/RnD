#include <cuda_runtime.h>
#include "cuda_ipc_manager.h"

__global__ void dummyDetection(unsigned char *frame, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        frame[idx] = 255 - frame[idx];  // Dummy operation (invert grayscale)
    }
}

extern "C" void detectObjects(CudaIpcManager &ipcManager, int width, int height) {
    void *d_frame = ipcManager.importMemory(WRITE);
    if (!d_frame) return;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    dummyDetection<<<gridSize, blockSize>>>((unsigned char*)d_frame, width, height);
}
