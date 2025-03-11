import cv2
import cupy as cp
import numpy as np
import multiprocessing
import time

def main(shared_mem_name):
    # Load image using OpenCV
    image = cv2.imread("nebula.jpeg", cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error: Cannot load image!")

    # Convert image to CuPy GPU array
    gpu_image = cp.asarray(image, dtype=cp.uint8)
    
    # Verify Shape
    print(f"Image shape: {gpu_image.shape}, dtype: {gpu_image.dtype}")

    # Get CUDA memory handle for IPC
    mem_handle = cp.cuda.runtime.ipcGetMemHandle(gpu_image.data.ptr)

    # Write memory handle to shared memory
    shared_mem = multiprocessing.shared_memory.SharedMemory(name=shared_mem_name, create=True, size=len(mem_handle))
    shared_mem.buf[:len(mem_handle)] = mem_handle

    print("Main: Image loaded, sending signal to median filter...")

    # Signal median filter process
    median_ready.set()
    time.sleep(10)

    # Cleanup shared memory
    shared_mem.close()
    shared_mem.unlink()

if __name__ == "__main__":
    shared_mem_name = "cuda_ipc_handle"
    median_ready = multiprocessing.Event()
    median_process = multiprocessing.Process(target=main, args=(shared_mem_name,))
    median_process.start()
    median_process.join()
