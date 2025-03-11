import cupy as cp
import multiprocessing
import time
from multiprocessing.shared_memory import SharedMemory

def median_filter(shared_mem_name):
    print("Median Filter: Waiting for signal...")
    time.sleep(2)

    # Open shared memory
    shared_mem = SharedMemory(name=shared_mem_name)
    mem_handle = bytes(shared_mem.buf[:])  # Read memory handle from shared memory
    shared_mem.close()

    # Open CUDA memory using IPC
    mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle)
    if mem_ptr is None or mem_ptr == 0:
        raise RuntimeError("Failed to open memory handle!")

    # Map the memory to CuPy
    mem = cp.cuda.memory.BaseMemory(mem_ptr)
    gpu_image = cp.ndarray((225, 225, 3), dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(mem, 0))

    # Verify Shape
    print(f"Image shape: {gpu_image.shape}, dtype: {gpu_image.dtype}")

    # Apply Median Filter
    filtered_image = cp.median(gpu_image, axis=2).astype(cp.uint8)

    # Allocate new GPU memory for filtered image
    filtered_gpu_image = cp.array(filtered_image, copy=True)

    # Create shared memory for filtered image handle
    new_mem_handle = cp.cuda.runtime.ipcGetMemHandle(filtered_gpu_image.data.ptr)
    new_shared_mem = SharedMemory(name="filtered_ipc_handle", create=True, size=len(new_mem_handle))
    new_shared_mem.buf[:len(new_mem_handle)] = new_mem_handle
    new_shared_mem.close()

    print("Median Filter: Applied, signaling sharpen filter...")
    sharpen_ready.set()  # Signal the next process

if __name__ == "__main__":
    shared_mem_name = "cuda_ipc_handle"
    sharpen_ready = multiprocessing.Event()

    median_process = multiprocessing.Process(target=median_filter, args=(shared_mem_name,))
    median_process.start()
    median_process.join()
