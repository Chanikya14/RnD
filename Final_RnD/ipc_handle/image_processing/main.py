import cv2
import cupy as cp
import numpy as np
import multiprocessing
from multiprocessing import shared_memory
import time

def main(shared_mem_name):

    cp.cuda.Device(0).use()  # Ensure CUDA device is initialized

    image = cv2.imread("nebula.jpeg", cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error: Cannot load image!")

    print(image.shape)
    gpu_image = cp.asarray(image, dtype=cp.uint8)
    
    # Get CUDA IPC memory handle
    mem_handle = cp.cuda.runtime.ipcGetMemHandle(gpu_image.data.ptr)
    mem_handle_bytes = np.frombuffer(mem_handle, dtype=np.uint8)

    # Ensure correct handle size
    handle_size = len(mem_handle)

    # Create shared memory
    shared_mem = shared_memory.SharedMemory(name=shared_mem_name, create=True, size=handle_size + 12)

    # Store handle
    shared_mem.buf[:handle_size] = mem_handle_bytes

    # Store shape
    np.ndarray(3, dtype=np.int32, buffer=shared_mem.buf, offset=handle_size)[:] = gpu_image.shape
    # print(gpu_image.shape.prod())

    print("Main: Image loaded, waiting for median filter to process...")

    time.sleep(5)  # Give the child process enough time

    shared_mem.close()
    shared_mem.unlink()

if __name__ == "__main__":
    shared_mem_name = "cuda_ipc_handle"
    median_process = multiprocessing.Process(target=main, args=(shared_mem_name,))
    median_process.start()
    median_process.join()
