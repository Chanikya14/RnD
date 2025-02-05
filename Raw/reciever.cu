#include <cuda_runtime.h>
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <thread>

#define SIZE 1024
#define SHM_NAME "/cuda_shm"

// CUDA Kernel to perform simple operations on data
__global__ void k2(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < SIZE) {
        data[idx] += 10; // Increment each element by its index
    }
}

int main() {
    int *d_data;
    cudaIpcMemHandle_t ipcHandle;
    bool handleReceived = false;

    // Open the shared memory where the IPC handle is stored
    int shm_fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory" << std::endl;
        return -1;
    }

    // Wait for the handle to be available for 10 seconds (check every second)
    auto start = std::chrono::steady_clock::now();
    while (true) {
        // Map the shared memory into the process's address space
        void *shm_ptr = mmap(NULL, sizeof(cudaIpcMemHandle_t), PROT_READ, MAP_SHARED, shm_fd, 0);
        if (shm_ptr != MAP_FAILED) {
            // Copy the IPC handle from shared memory
            memcpy(&ipcHandle, shm_ptr, sizeof(cudaIpcMemHandle_t));
            std::cout << "Received IPC Handle from shared memory" << std::endl;
            handleReceived = true;
            break; // Exit the loop once the handle is received
        }

        // Check if 10 seconds have passed
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - start;
        if (elapsed.count() >= 10) {
            std::cerr << "Failed to receive IPC handle after 10 seconds" << std::endl;
            return -1; // Exit after 10 seconds without receiving the handle
        }

        // Wait for 1 second before trying again
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Open the memory using the IPC handle
    if (handleReceived) {
        cudaError_t err = cudaIpcOpenMemHandle((void**)&d_data, ipcHandle, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
            std::cerr << "cudaIpcOpenMemHandle failed: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

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

    return 0;
}
