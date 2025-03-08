import cv2
import cupy as cp
import numpy as np
import multiprocessing
import time

def main():
    # Load image using OpenCV
    image = cv2.imread("nebula.jpeg", cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error: Cannot load image!")

    # Convert image to CuPy GPU array
    gpu_image = cp.asarray(image, dtype=cp.uint8)
    # Verify Shape
    print(f"Image shape: {gpu_image.shape}, dtype: {gpu_image.dtype}")

    # Get CUDA memory handle for inter-process communication
    mem_handle = cp.cuda.runtime.ipcGetMemHandle(gpu_image.data.ptr)

    # Save memory handle to a file
    with open("mem_handle.bin", "wb") as f:
        f.write(mem_handle)

    print("Main: Image loaded, sending signal to median filter...")

    # Signal median filter process
    median_ready.set()
    time.sleep(10)

if __name__ == "__main__":
    median_ready = multiprocessing.Event()
    median_process = multiprocessing.Process(target=main)
    median_process.start()
    median_process.join()
