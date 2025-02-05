#include "Cuda_Ipc_Manager.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>

CudaIpcManager::CudaIpcManager(const std::string& name) : shmName(name), shm_fd(-1), shm_ptr(nullptr) {}

// Sender: Export IPC handle
bool CudaIpcManager::exportMemory(void* d_ptr, size_t size) {
    cudaIpcMemHandle_t ipcHandle;
    if (cudaIpcGetMemHandle(&ipcHandle, d_ptr) != cudaSuccess) {
        std::cerr << "Failed to get CUDA IPC handle\n";
        return false;
    }

    shm_fd = shm_open(shmName.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory\n";
        return false;
    }

    if (ftruncate(shm_fd, sizeof(cudaIpcMemHandle_t)) == -1) {
        std::cerr << "Failed to set shared memory size\n";
        return false;
    }

    shm_ptr = mmap(NULL, sizeof(cudaIpcMemHandle_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory\n";
        return false;
    }

    memcpy(shm_ptr, &ipcHandle, sizeof(cudaIpcMemHandle_t));
    std::cerr << "Sucess"<<std::endl;
    return true;
}

// Receiver: Import IPC handle
void* CudaIpcManager::importMemory() {

    cudaIpcMemHandle_t ipcHandle;
    bool handleReceived = false;
    void* d_data;

    // Open the shared memory where the IPC handle is stored
    int shm_fd = shm_open(shmName.c_str(), O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory" << std::endl;
        return NULL;
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
            return NULL; // Exit after 10 seconds without receiving the handle
        }

        // Wait for 1 second before trying again
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Open the memory using the IPC handle
    if (handleReceived) {
        cudaError_t err = cudaIpcOpenMemHandle((void**)(&d_data), ipcHandle, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
            std::cerr << "cudaIpcOpenMemHandle failed: " << cudaGetErrorString(err) << std::endl;
            return NULL;
        }
    }
    return d_data;
}

// Cleanup function
void CudaIpcManager::cleanup() {
    if (shm_ptr) {
        munmap(shm_ptr, sizeof(cudaIpcMemHandle_t));
        shm_ptr = nullptr;
    }
    if (shm_fd != -1) {
        close(shm_fd);
        shm_unlink(shmName.c_str());
        shm_fd = -1;
    }
}

CudaIpcManager::~CudaIpcManager() {
    cleanup();
}
