#ifndef CUDA_IPC_MANAGER_H
#define CUDA_IPC_MANAGER_H

#include <cuda_runtime.h>
#include <string>
#include <chrono>
#include <thread>

class CudaIpcManager {
public:
    CudaIpcManager(const std::string& shmName);
    ~CudaIpcManager();

    // Sender: Exports GPU memory handle
    bool exportMemory(void* d_ptr, size_t size);

    // Receiver: Imports GPU memory handle
    void* importMemory();

    // Cleanup
    void cleanup();

private:
    std::string shmName;
    int shm_fd;
    void* shm_ptr;
};

#endif // CUDA_IPC_MANAGER_H
