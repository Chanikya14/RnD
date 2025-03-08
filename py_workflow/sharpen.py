import cupy as cp
import numpy as np
import multiprocessing

def sharpen_filter():
    # Wait for the signal from median.py
    # sharpen_ready.wait()
    # time.sleep(5)

    # Load memory handle
    with open("median_handle.bin", "rb") as f:
        mem_handle = f.read()

    # Restore GPU memory
    gpu_image = cp.cuda.runtime.ipcOpenMemHandle(mem_handle)
    image = cp.ndarray((512, 512, 3), dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(gpu_image, 0))

    # Sharpening kernel
    sharpen_kernel = cp.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=cp.float32)

    # Apply convolution
    sharpened_image = cp.apply_along_axis(lambda x: cp.convolve(x, sharpen_kernel, mode='same'), 2, image)

    # Save new handle
    new_mem_handle = cp.cuda.runtime.ipcGetMemHandle(sharpened_image.data.ptr)
    with open("sharpen_handle.bin", "wb") as f:
        f.write(new_mem_handle)

    print("Sharpen Filter: Applied, signaling thresholding...")

    # Signal the threshold process
    threshold_ready.set()

if __name__ == "__main__":
    sharpen_ready = multiprocessing.Event()
    threshold_ready = multiprocessing.Event()

    sharpen_process = multiprocessing.Process(target=sharpen_filter)
    sharpen_process.start()
    sharpen_process.join()
