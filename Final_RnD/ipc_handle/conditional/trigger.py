import cupy as cp
import numpy as np
from multiprocessing import shared_memory
import time
import cv2
import torch
import torch.nn.functional as F
from multiprocessing import Process


def trigger_cat_workflow(input_mem_name, flag_mem):
    print("→ CAT workflow started in a separate process")

    existing_shm = shared_memory.SharedMemory(name=flag_mem)
    flag = np.ndarray((1,), dtype=np.uint8, buffer=existing_shm.buf)

    if flag[0] != 1:
        return

    input_mem = shared_memory.SharedMemory(name=input_mem_name)

    handle_size = 64
    handle = input_mem.buf[:handle_size]
    mem_handle_bytes = bytes(handle)
    img_shape = np.ndarray(3, dtype=np.int32, buffer=input_mem.buf, offset=handle_size)

    print("cat workflow: Shape received", img_shape)

    cp.cuda.Device(0).use()
    mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle_bytes)
    mem = cp.cuda.memory.MemoryPointer(
        cp.cuda.memory.UnownedMemory(mem_ptr, img_shape.prod() * cp.uint8().itemsize, cp.cuda.Device(0)), 0
    )
    # Convert to CuPy array
    gpu_image = cp.ndarray(tuple(img_shape), dtype=cp.uint8, memptr=mem)

    # Convert to float32 on GPU
    img_f32 = gpu_image.astype(cp.float32)

    # Apply grayscale (luminosity method)
    gray = 0.299 * img_f32[:, :, 0] + 0.587 * img_f32[:, :, 1] + 0.114 * img_f32[:, :, 2]

    # Convert back to uint8
    gray_u8 = gray.astype(cp.uint8)

    # Expand to 3 channels so it looks like a grayscale RGB image
    gray_3ch = cp.stack([gray_u8] * 3, axis=-1)

    # Move from GPU to CPU for saving
    cpu_image = cp.asnumpy(gray_3ch)

    # Save to file using OpenCV
    cv2.imwrite('grayscale_output.png', cpu_image)

def trigger_dog_workflow(input_mem_name, flag_mem):
    print("→ DOG workflow started in a separate process")

    existing_shm = shared_memory.SharedMemory(name=flag_mem)
    flag = np.ndarray((1,), dtype=np.uint8, buffer=existing_shm.buf)

    if flag[0] != 2:
        return

    input_mem = shared_memory.SharedMemory(name=input_mem_name)

    handle_size = 64
    handle = input_mem.buf[:handle_size]
    mem_handle_bytes = bytes(handle)
    img_shape = np.ndarray(3, dtype=np.int32, buffer=input_mem.buf, offset=handle_size)

    print("dog workflow: Shape received", img_shape)

    cp.cuda.Device(0).use()
    mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle_bytes)
    mem = cp.cuda.memory.MemoryPointer(
        cp.cuda.memory.UnownedMemory(mem_ptr, img_shape.prod() * cp.uint8().itemsize, cp.cuda.Device(0)), 0
    )
    # Convert to CuPy array
    gpu_image = cp.ndarray(tuple(img_shape), dtype=cp.uint8, memptr=mem)

    gpu_inverted = 255 - gpu_image

    # Move from GPU to CPU for saving
    cpu_image = cp.asnumpy(gpu_inverted)

    # Save to file using OpenCV
    cv2.imwrite('inverted_output.png', cpu_image)


if __name__ == "__main__":
    input_mem_name = "handle1"
    flag_mem = "flag"
    time.sleep(5)

    start = time.time()
    trigger_cat_workflow(input_mem_name, flag_mem)
    trigger_dog_workflow(input_mem_name, flag_mem)

    end = time.time()
    elapsed_time = end - start
    print(f"trigger Elapsed time: {elapsed_time:.2f} seconds")

