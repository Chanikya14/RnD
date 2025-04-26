import cv2
import cupy as cp
import numpy as np
import multiprocessing
from multiprocessing import shared_memory
import time


def main(shared_mem_name):
    # print("sgs")
    # Read image using OpenCV

    start = time.time()
    img = cv2.imread('dog1.jpeg', cv2.IMREAD_COLOR)
    gpu_image = cp.asarray(img, dtype=cp.uint8)
    mem_handle = cp.cuda.runtime.ipcGetMemHandle(gpu_image.data.ptr)
    mem_handle_bytes = np.frombuffer(mem_handle, dtype=np.uint8)

    handle_size = len(mem_handle)
    # print("sdf")
    shared_mem = shared_memory.SharedMemory(name=shared_mem_name, create=True, size=handle_size + 12)
    shared_mem.buf[:handle_size] = mem_handle_bytes
    np.ndarray(3, dtype=np.int32, buffer=shared_mem.buf, offset=handle_size)[:] = gpu_image.shape

    end = time.time()
    elapsed_time = end - start
    print(f"start Elapsed time: {elapsed_time:.2f} seconds")

    print("Main: Image loaded")
    print("====================END===================")
    time.sleep(10)
    shared_mem.close()
    shared_mem.unlink()
if __name__ == "__main__":
    shared_mem_name = "handle1"
    start = multiprocessing.Process(target=main, args=(shared_mem_name,))
    start.start()
    start.join()

