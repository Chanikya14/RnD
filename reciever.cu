#include <cuda_runtime.h>
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "Cuda_Ipc_Manager.h"

#define SIZE 1024
#define SHM_NAME "/cuda_shm"
#define META_SHM "/meta_shm"

// CUDA Kernel to perform simple operations on data
__global__ void k2(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < SIZE) {
        data[idx] = 120; // Increment each element by its index
    }
}

int main() {
    
    CudaIpcManager manager(SHM_NAME); // Object creation
    int *d_data = (int*)manager.importMemory(); // import handle

    if(d_data!=NULL) {
        // Launch kernel to modify data
        int threadsPerBlock = 256;
        int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
        k2<<<blocksPerGrid, threadsPerBlock>>>(d_data);
        cudaDeviceSynchronize();

        // Copy modified data back to host
        int* h_data = new int[SIZE];
        cudaMemcpy(h_data, d_data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Modified first element in shared memory: " << h_data[0] << std::endl;

        // Clean up
        cudaFree(d_data);
        delete[] h_data;
    }
    else {
        std::cerr << "Import memory failed!!"<<std::endl;
    }

    return 0;
}
