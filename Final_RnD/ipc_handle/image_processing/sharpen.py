import cupy as cp
import numpy as np
from multiprocessing import shared_memory
import time
import cupyx.scipy.ndimage as cnd  # Import CuPy's optimized convolution function

def sharpen_filter(input_mem_name, output_mem_name):
    print("Sharpen Filter: Waiting for shared memory...")
    start = time.time()
    # Open input shared memory
    input_mem = shared_memory.SharedMemory(name=input_mem_name)

    # Read image shape
    img_shape = np.ndarray(3, dtype=np.int32, buffer=input_mem.buf, offset=0)
    
    # print("Sharpen Filter: Shape received", img_shape)

    # Read image data
    img_data = np.frombuffer(input_mem.buf, dtype=np.uint8, offset=12).reshape(tuple(img_shape))
    gpu_image = cp.asarray(img_data)

    # print("Sharpen Filter: Image received, applying sharpen filter...")

    # **Sharpen Kernel (3x3)**
    kernel = cp.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=cp.float32)

    # **Apply Sharpen Filter**
    sharpened_image = cp.empty_like(gpu_image)
    for c in range(gpu_image.shape[2]):  # Apply per channel
        sharpened_image[:, :, c] = cp.clip(cnd.convolve(gpu_image[:, :, c], kernel, mode='constant'), 0, 255)

    # print("Sharpen Filter: Applied successfully!")

    # **Write to Shared Memory**
    output_size = sharpened_image.nbytes + 12  # 12 bytes for shape storage
    output_mem = shared_memory.SharedMemory(name=output_mem_name, create=True, size=output_size)

    # Store shape
    np.ndarray(3, dtype=np.int32, buffer=output_mem.buf, offset=0)[:] = img_shape

    # Store sharpened image
    output_mem.buf[12:] = sharpened_image.get().tobytes()

    end = time.time()
    print(f"sharpen Elapsed: {end - start:.4f} seconds")

    print("Sharpen Filter: Output written to shared memory.")

    # Wait before cleanup
    time.sleep(5)

    # Cleanup
    input_mem.close()
    output_mem.close()

if __name__ == "__main__":
    input_mem_name = "median_output"  # Read from median filter output
    output_mem_name = "sharpen_output"  # Write to threshold filter input
    sharpen_filter(input_mem_name, output_mem_name)
