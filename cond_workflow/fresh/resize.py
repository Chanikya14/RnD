import cupy as cp
import numpy as np
from multiprocessing import shared_memory
import time
import cv2
import torch
from cupyx.scipy.ndimage import zoom


def check(gpu_image):
    cpu_image = gpu_image.get()  # shape: (H, W, 3), dtype: uint8

    print(cpu_image.shape)
    # Step 3: Display or save using OpenCV
    cv2.imshow('Recovered Image', cpu_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cupy_to_torch(cp_array):
    return torch.from_dlpack(cp_array.toDlpack())

def resize(input_mem_name):
    input_mem = shared_memory.SharedMemory(name=input_mem_name)

    handle_size = 64
    handle = input_mem.buf[:handle_size]
    mem_handle_bytes = bytes(handle)
    img_shape = np.ndarray(3, dtype=np.int32, buffer=input_mem.buf, offset=handle_size)

    print("resize: Shape received", img_shape)

    cp.cuda.Device(0).use()
    mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle_bytes)
    mem = cp.cuda.memory.MemoryPointer(
        cp.cuda.memory.UnownedMemory(mem_ptr, img_shape.prod() * cp.uint8().itemsize, cp.cuda.Device(0)), 0
    )

    # Convert to CuPy array
    gpu_image = cp.ndarray(tuple(img_shape), dtype=cp.uint8, memptr=mem)
    # print(gpu_image)
    a, b, c = (gpu_image.shape)

    zoom_factors = (
        0.9,  # height scale
        0.9,  # width scale
        1                         # no scale on channel dim
    )

    # Resize on GPU
    resized = zoom(gpu_image, zoom_factors, order=1)
    a, b, c = resized.shape
    # check(resized)
    gpu_image[:a, :b, :c] = resized
    gpu_image[a:, :, :] = 0
    gpu_image[:, b:, :] = 0

    # np.ndarray(3, dtype=np.int32, buffer=input_mem.buf, offset=handle_size)[:] = resized.shape

    print("✅ Resized on GPU:", resized.shape)
    print("====================END===================")

    cp.cuda.runtime.ipcCloseMemHandle(mem_ptr)
    time.sleep(20)


    input_mem.close()

if __name__ == "__main__":
    input_mem_name = "handle1"
    resize(input_mem_name)