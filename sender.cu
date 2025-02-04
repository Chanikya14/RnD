#include <cuda_runtime.h>
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>

#define SIZE 1024
#define SHM_NAME "/cuda_shm"

__global__ void k1(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        data[0] = 45;
        printf("data[0] = %d\n", data[0]);  // Print the value of data[0]
    }
}

int main() {
    int *d_data;
    cudaIpcMemHandle_t ipcHandle;

    // Allocate memory on the GPU
    cudaMalloc(&d_data, SIZE * sizeof(int));

    k1<<<1,1>>> (d_data);

    cudaDeviceSynchronize();
    // Get the IPC memory handle for sharing
    cudaError_t err = cudaIpcGetMemHandle(&ipcHandle, d_data);
    if (err != cudaSuccess) {
        std::cerr << "cudaIpcGetMemHandle failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Open or create shared memory
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory" << std::endl;
        return -1;
    }

    // Set the size of the shared memory
    if (ftruncate(shm_fd, sizeof(cudaIpcMemHandle_t)) == -1) {
        std::cerr << "Failed to set shared memory size" << std::endl;
        return -1;
    }

    // Map the shared memory into the process's address space
    void *shm_ptr = mmap(NULL, sizeof(cudaIpcMemHandle_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory" << std::endl;
        return -1;
    }

    // Copy the IPC handle to shared memory
    memcpy(shm_ptr, &ipcHandle, sizeof(cudaIpcMemHandle_t));

    std::cout << "IPC Handle sent to shared memory" << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Clean up (In a real-world app, this would be done after receiving in Process B)
    shm_unlink(SHM_NAME);  // Unlink shared memory object

    cudaFree(d_data);
    return 0;
}
