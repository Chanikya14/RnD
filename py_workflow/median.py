import cupy as cp
import numpy as np
import multiprocessing
import time

def median_filter():
    print("Median Filter: Waiting for signal...")
    # median_ready.wait()  # Wait for signal from main process
    time.sleep(2)

    # Load memory handle
    with open("mem_handle.bin", "rb") as f:
        mem_handle = f.read()

    # Open shared memory (Use DeviceMemory instead of UnifiedMemory)
    mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle)

    if mem_ptr is None or mem_ptr == 0:
        raise RuntimeError("Failed to open memory handle!")

    mem = cp.cuda.memory.BaseMemory(mem_ptr)  # Correct type for CuPy memory
    gpu_image = cp.ndarray((225, 225, 3), dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(mem, 0))
    
    # Verify Shape
    print(f"Image shape: {gpu_image.shape}, dtype: {gpu_image.dtype}")
    
    print("after ")
    # return

    print(gpu_image[0][0][2])
    filtered_image = cp.median(gpu_image, axis=2).astype(cp.uint8)

    # Ensure output is valid
    if filtered_image is None:
        raise RuntimeError("Filtered image is invalid!")

    return

    
    # Apply Median Filter
    filtered_image = cp.median(gpu_image, axis=2)
    return

    print("kasdjf")

    # Allocate new GPU memory for filtered image
    filtered_gpu_image = cp.array(filtered_image, copy=True)  # Creates a new GPU memory

    print("2")

    # Save new memory handle
    new_mem_handle = cp.cuda.runtime.ipcGetMemHandle(filtered_gpu_image.data.ptr)
    with open("median_handle.bin", "wb") as f:
        f.write(new_mem_handle)

    print("Median Filter: Applied, signaling sharpen filter...")
    sharpen_ready.set()  # Signal the next process

if __name__ == "__main__":
    median_ready = multiprocessing.Event()
    sharpen_ready = multiprocessing.Event()

    median_process = multiprocessing.Process(target=median_filter)
    median_process.start()
    median_process.join()
