import cupy as cp
import numpy as np
from multiprocessing import shared_memory
import time

def median_filter(input_mem_name, output_mem_name):
    print("Median Filter: Waiting for shared memory...")

    # Open shared memory for input
    input_mem = shared_memory.SharedMemory(name=input_mem_name)

    # Read memory handle
    handle_size = 64  # Ensure correct IPC handle size
    mem_handle_bytes = bytes(input_mem.buf[:handle_size])

    # Read image shape
    img_shape = np.ndarray(3, dtype=np.int32, buffer=input_mem.buf, offset=handle_size)

    print("Median Filter: Shape received", img_shape)

    # Open CUDA memory using IPC
    cp.cuda.Device(0).use()  # Ensure CUDA is initialized
    mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle_bytes)

    # Create CuPy MemoryPointer
    mem = cp.cuda.memory.MemoryPointer(
        cp.cuda.memory.UnownedMemory(mem_ptr, img_shape.prod() * cp.uint8().itemsize, cp.cuda.Device(0)), 0
    )

    # Convert to CuPy array
    gpu_image = cp.ndarray(tuple(img_shape), dtype=cp.uint8, memptr=mem)

    print("Median Filter: Image received, applying median filter...")

    # **Apply Median Filter**
    pad_img = cp.pad(gpu_image, ((2, 2), (2, 2), (0, 0)), mode='reflect')  # Padding for 5x5 kernel
    window_shape = (5, 5)

    # Extract sliding windows for each channel separately
    strided_img = cp.lib.stride_tricks.sliding_window_view(pad_img, window_shape, axis=(0, 1))

    # Compute median across the last two axes (5x5 window)
    filtered_image = cp.median(strided_img, axis=(-2, -1)).astype(cp.uint8)

    print("Median Filter: Applied filter successfully! Writing to shared memory...")

    # **Store filtered image in new shared memory**
    filtered_mem = shared_memory.SharedMemory(name=output_mem_name, create=True, size=filtered_image.nbytes + 12)

    # Write image shape
    np.ndarray(3, dtype=np.int32, buffer=filtered_mem.buf, offset=0)[:] = img_shape

    # Write image data
    filtered_mem.buf[12:] = filtered_image.ravel().tobytes()

    # Wait before cleanup
    time.sleep(10)

    # Cleanup
    cp.cuda.runtime.ipcCloseMemHandle(mem_ptr)
    input_mem.close()
    filtered_mem.close()

if __name__ == "__main__":
    input_mem_name = "cuda_ipc_handle"
    output_mem_name = "median_output"
    median_filter(input_mem_name, output_mem_name)
