#ifndef CUDA_IPC_MANAGER_H
#define CUDA_IPC_MANAGER_H

#include <cuda_runtime.h>
#include <string>
#include <chrono>
#include <thread>

enum AccessMode { READ, WRITE };

class CudaIpcManager {
private:
    std::string data_shm, meta_shm;
    int shm_fd, data_fd;
    void* shm_ptr;
public:
    CudaIpcManager(const std::string& meta_shm, const std::string& data_shm);
    ~CudaIpcManager();

    // Sender: Exports GPU memory handle
    bool exportMemory(void* d_ptr, size_t size);

    // Receiver: Imports GPU memory handle
    void* importMemory(AccessMode mode);

    // Cleanup
    void cleanup();
};

#endif // CUDA_IPC_MANAGER_H
