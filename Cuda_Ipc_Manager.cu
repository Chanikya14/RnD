#include "Cuda_Ipc_Manager.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <pthread.h>

enum AccessMode { READ, WRITE };

struct Control_data {
    size_t offset_start;
    size_t offset_end;
    pthread_rwlock_t rwlock; // Reader-Writer Lock

    Control_data() {
        pthread_rwlock_init(&rwlock, NULL);
    }

    ~Control_data() {
        pthread_rwlock_destroy(&rwlock);
    }

    bool acquireReadLock() {
        return pthread_rwlock_rdlock(&rwlock) == 0;
    }

    bool acquireWriteLock() {
        return pthread_rwlock_wrlock(&rwlock) == 0;
    }

    void releaseLock() {
        pthread_rwlock_unlock(&rwlock);
    }
};

class CudaIpcManager {
private:
    std::string data_shm, meta_shm;
    int shm_fd;
    void* shm_ptr;

public:
    CudaIpcManager(const std::string& data_name, const std::string& meta_name);
    bool exportMemory(void* d_ptr, size_t size);
    const void* importMemory(AccessMode mode);
    void cleanup();
    ~CudaIpcManager();
};

// Constructor
CudaIpcManager::CudaIpcManager(const std::string& data_name, const std::string& meta_name) : data_shm(data_name), meta_shm(meta_name), shm_fd(-1), shm_ptr(nullptr) {}

// Sender: Export IPC handle (Checks and acquires write lock)
bool CudaIpcManager::exportMemory(void* d_ptr, size_t size) {
    shm_fd = shm_open(meta_shm.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory\n";
        return false;
    }

    if (ftruncate(shm_fd, sizeof(Control_data) + sizeof(cudaIpcMemHandle_t)) == -1) {
        std::cerr << "Failed to set shared memory size\n";
        return false;
    }

    shm_ptr = mmap(NULL, sizeof(Control_data) + sizeof(cudaIpcMemHandle_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory\n";
        return false;
    }

    Control_data* ctrl = static_cast<Control_data*>(shm_ptr);
    if (!ctrl->acquireWriteLock()) {
        std::cerr << "Failed to acquire write lock\n";
        return false;
    }

    cudaIpcMemHandle_t ipcHandle;
    if (cudaIpcGetMemHandle(&ipcHandle, d_ptr) != cudaSuccess) {
        std::cerr << "Failed to get CUDA IPC handle\n";
        ctrl->releaseLock();
        return false;
    }

    memcpy(static_cast<void*>(ctrl + 1), &ipcHandle, sizeof(cudaIpcMemHandle_t));

    std::cerr << "Export success" << std::endl;
    ctrl->releaseLock();
    return true;
}

// Receiver: Import IPC handle (Checks and acquires read/write lock)
const void* CudaIpcManager::importMemory(AccessMode mode) {
    shm_fd = shm_open(shmName.c_str(), O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory" << std::endl;
        return NULL;
    }

    shm_ptr = mmap(NULL, sizeof(Control_data) + sizeof(cudaIpcMemHandle_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory\n";
        return NULL;
    }

    Control_data* ctrl = static_cast<Control_data*>(shm_ptr);
    if (mode == READ) {
        if (!ctrl->acquireReadLock()) {
            std::cerr << "Failed to acquire read lock\n";
            return NULL;
        }
    } else {
        if (!ctrl->acquireWriteLock()) {
            std::cerr << "Failed to acquire write lock\n";
            return NULL;
        }
    }

    cudaIpcMemHandle_t ipcHandle;
    memcpy(&ipcHandle, static_cast<void*>(ctrl + 1), sizeof(cudaIpcMemHandle_t));

    void* d_data = nullptr;
    if (cudaIpcOpenMemHandle(&d_data, ipcHandle, cudaIpcMemLazyEnablePeerAccess) != cudaSuccess) {
        std::cerr << "cudaIpcOpenMemHandle failed\n";
        ctrl->releaseLock();
        return NULL;
    }

    ctrl->releaseLock();
    return d_data;
}

// Cleanup function
void CudaIpcManager::cleanup() {
    if (shm_ptr) {
        munmap(shm_ptr, sizeof(Control_data) + sizeof(cudaIpcMemHandle_t));
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
