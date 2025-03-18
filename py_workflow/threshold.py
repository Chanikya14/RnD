import cupy as cp
import numpy as np
import cv2
from multiprocessing import shared_memory

def threshold_filter(input_mem_name, output_filename, threshold_value=128):
    print("Threshold Filter: Waiting for shared memory...")

    # Open input shared memory
    input_mem = shared_memory.SharedMemory(name=input_mem_name)

    # Read image shape
    img_shape = np.ndarray(3, dtype=np.int32, buffer=input_mem.buf, offset=0)

    print("Threshold Filter: Shape received", img_shape)

    # Read image data
    img_data = np.frombuffer(input_mem.buf, dtype=np.uint8, offset=12).reshape(tuple(img_shape))
    gpu_image = cp.asarray(img_data)

    print("Threshold Filter: Applying threshold filter...")

    # **Apply Threshold**
    gpu_thresholded = cp.where(gpu_image > threshold_value, 255, 0).astype(cp.uint8)

    # Convert back to NumPy for saving
    final_image = cp.asnumpy(gpu_thresholded)

    # **Save Image**
    cv2.imwrite(output_filename, final_image)

    print(f"Threshold Filter: Image saved as {output_filename}")

    # Cleanup
    input_mem.close()

if __name__ == "__main__":
    input_mem_name = "sharpen_output"
    output_filename = "final_output.png"
    threshold_filter(input_mem_name, output_filename)
