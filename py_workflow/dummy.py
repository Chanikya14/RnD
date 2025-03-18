import cupy as cp
import numpy as np
import multiprocessing
from multiprocessing import shared_memory
import time

def create_shared_memory(shared_mem_name):
    cp.cuda.Device(0).use()  # Ensure CUDA device is initialized
    gpu_data = cp.zeros((100, 100), dtype=cp.float32)  # Example GPU data
    mem_handle = cp.cuda.runtime.ipcGetMemHandle(gpu_data.data.ptr)  # Get IPC memory handle
    handle_bytes = np.frombuffer(mem_handle, dtype=np.uint8)  # Convert handle to bytes

    shared_mem = shared_memory.SharedMemory(name=shared_mem_name, create=True, size=handle_bytes.nbytes)
    shared_mem.buf[: handle_bytes.nbytes] = handle_bytes  # Store handle in shared memory

    print("Memory handle written to shared memory")

    time.sleep(10)  # Wait to ensure the second process reads it
    shared_mem.close()
    shared_mem.unlink()

def read_shared_memory(shared_mem_name):
    time.sleep(2)  # Ensure shared memory is created first
    cp.cuda.Device(0).use()  # Ensure CUDA device is initialized

    shared_mem = shared_memory.SharedMemory(name=shared_mem_name)
    handle_bytes = bytes(shared_mem.buf[:64])  # Read handle from shared memory

    # Open shared memory handle
    mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(handle_bytes)

    # Wrap raw pointer using CuPy's MemoryPointer
    mem_obj = cp.cuda.UnownedMemory(mem_ptr, 100 * 100 * 4, owner=None)
    gpu_data = cp.ndarray((100, 100), dtype=cp.float32, memptr=cp.cuda.MemoryPointer(mem_obj, 0))

    print("Accessed data from shared memory:", gpu_data)

    shared_mem.close()

if __name__ == "__main__":
    shared_mem_name = "cuda_ipc_handle"
    
    p1 = multiprocessing.Process(target=create_shared_memory, args=(shared_mem_name,))
    p2 = multiprocessing.Process(target=read_shared_memory, args=(shared_mem_name,))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
