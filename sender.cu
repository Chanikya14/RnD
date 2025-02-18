#include <cuda_runtime.h>
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include "Cuda_Ipc_Manager.h"

#define SIZE 1024
#define SHM_NAME "/cuda_shm"
#define META_SHM "/meta_shm"

__global__ void k1(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        data[0] = 45;
        printf("data[0] = %d\n", data[0]);  // Print the value of data[0]
    }
}

int main() {

    CudaIpcManager manager(SHM_NAME, META_SHM); // Object creation

    int *d_data;
    cudaMalloc(&d_data, SIZE * sizeof(int));

    k1<<<1,1>>> (d_data);
    cudaDeviceSynchronize();

    manager.exportMemory(d_data, SIZE * sizeof(int)); // export handle

    std::cout << "IPC Handle sent to shared memory" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));

    manager.cleanup();

    cudaFree(d_data);
    return 0;
}
