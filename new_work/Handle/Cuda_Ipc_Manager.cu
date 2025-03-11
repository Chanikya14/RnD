#include "Cuda_Ipc_Manager.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <pthread.h>


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

// Constructor
CudaIpcManager::CudaIpcManager(const std::string& data_name, const std::string& meta_name) : data_shm(data_name), meta_shm(meta_name), shm_fd(-1), shm_ptr(nullptr) {}

// Sender: Export IPC handle (Checks and acquires write lock)
bool CudaIpcManager::exportMemory(void* d_ptr, size_t size) {
    shm_fd = shm_open(meta_shm.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory\n";
        return false;
    }

    if (ftruncate(shm_fd, sizeof(Control_data)) == -1) {
        std::cerr << "Failed to set shared memory size for control data\n";
        return false;
    }

    shm_ptr = mmap(NULL, sizeof(Control_data), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory for control data\n";
        return false;
    }

    Control_data* ctrl = static_cast<Control_data*>(shm_ptr);
    if (!ctrl->acquireWriteLock()) {
        std::cerr << "Failed to acquire write lock\n";
        return false;
    }
    // Acquired write lock

    cudaIpcMemHandle_t ipcHandle;
    if (cudaIpcGetMemHandle(&ipcHandle, d_ptr) != cudaSuccess) {
        std::cerr << "Failed to get CUDA IPC handle\n";
        ctrl->releaseLock();
        return false;
    }


    data_fd = shm_open(data_shm.c_str(), O_CREAT | O_RDWR, 0666);
    if (data_fd == -1) {
        std::cerr << "Failed to open shared memory for memory handle\n";
        return false;
    }

    if (ftruncate(data_fd, sizeof(cudaIpcMemHandle_t)) == -1) {
        std::cerr << "Failed to set shared memory size for memory handle\n";
        return false;
    }

    shm_ptr = mmap(NULL, sizeof(cudaIpcMemHandle_t), PROT_READ | PROT_WRITE, MAP_SHARED, data_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory for memory handle\n";
        return false;
    }

    memcpy(shm_ptr, &ipcHandle, sizeof(cudaIpcMemHandle_t));
    ctrl->releaseLock();
    return true;
}

// Receiver: Import IPC handle (Checks and acquires read/write lock)
void* CudaIpcManager::importMemory(AccessMode mode) {
    shm_fd = shm_open(meta_shm.c_str(), O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory" << std::endl;
        return nullptr;
    }

    shm_ptr = mmap(NULL, sizeof(Control_data), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory\n";
        return nullptr;
    }

    Control_data* ctrl = static_cast<Control_data*>(shm_ptr);
    if (mode == READ) {
        if (!ctrl->acquireReadLock()) {
            std::cerr << "Failed to acquire read lock\n";
            munmap(shm_ptr, sizeof(Control_data)); // Unmap memory before returning
            close(shm_fd);
            return nullptr;
        }
    } else {
        if (!ctrl->acquireWriteLock()) {
            std::cerr << "Failed to acquire write lock\n";
            munmap(shm_ptr, sizeof(Control_data));
            close(shm_fd);
            return nullptr;
        }
    }


    bool handleReceived = false;
    shm_fd = shm_open(data_shm.c_str(), O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory" << std::endl;
        ctrl->releaseLock();
        munmap(shm_ptr, sizeof(Control_data));
        close(shm_fd);
        return NULL;
    }

    cudaIpcMemHandle_t ipcHandle;
    // Wait for the handle to be available for 10 seconds (check every second)
    auto start = std::chrono::steady_clock::now();
    while (true) {
        // Map the shared memory into the process's address space
        void *handle_ptr = mmap(NULL, sizeof(cudaIpcMemHandle_t), PROT_READ, MAP_SHARED, shm_fd, 0);
        if (handle_ptr != MAP_FAILED) {
            // Copy the IPC handle from shared memory
            memcpy(&ipcHandle, handle_ptr, sizeof(cudaIpcMemHandle_t));
            munmap(handle_ptr, sizeof(cudaIpcMemHandle_t));  // Free memory immediately
            std::cout << "Received IPC Handle from shared memory" << std::endl;
            handleReceived = true;
            break; // Exit the loop once the handle is received
        }

        // Check if 10 seconds have passed
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - start;
        if (elapsed.count() >= 10) {
            std::cerr << "Failed to receive IPC handle after 10 seconds" << std::endl;
            ctrl->releaseLock();
            munmap(shm_ptr, sizeof(Control_data));
            close(shm_fd);
            return NULL; // Exit after 10 seconds without receiving the handle
        }

        // Wait for 1 second before trying again
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    void *d_data;
    // Open the memory using the IPC handle
    if (handleReceived) {
        cudaError_t err = cudaIpcOpenMemHandle((void**)(&d_data), ipcHandle, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
            std::cerr << "cudaIpcOpenMemHandle failed: " << cudaGetErrorString(err) << std::endl;
            ctrl->releaseLock();
            munmap(shm_ptr, sizeof(Control_data));
            close(shm_fd);
            return NULL;
        }
    }
    ctrl->releaseLock();
    munmap(shm_ptr, sizeof(Control_data)); // Cleanup before returning
    close(shm_fd);
    
    // Enforce const correctness for read mode
    // return (mode == READ) ? static_cast<const void*>(d_data) : d_data;
    return d_data;
}

void CudaIpcManager::release_RL(){

}


// Cleanup function
void CudaIpcManager::cleanup() {
    // Unmap shared memory for control data
    if (shm_ptr) {
        munmap(shm_ptr, sizeof(Control_data)); // Unmap shared memory
        shm_ptr = nullptr;
    }

    // Unmap shared memory for IPC handle if allocated
    // if (handle_ptr) {
    //     munmap(handle_ptr, sizeof(cudaIpcMemHandle_t)); // Unmap the handle
    //     handle_ptr = nullptr;
    // }

    // Close shared memory file descriptor
    if (shm_fd != -1) {
        close(shm_fd);
        shm_unlink(meta_shm.c_str()); // Unlink shared memory if necessary
        shm_fd = -1;
    }
}


CudaIpcManager::~CudaIpcManager() {
    cleanup();
}
