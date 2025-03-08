import cupy as cp
import cv2
import multiprocessing

def threshold_filter():
    # Wait for the signal from sharpen.py
    # threshold_ready.wait()
    # time.sleep(5)

    # Load memory handle
    with open("sharpen_handle.bin", "rb") as f:
        mem_handle = f.read()

    # Restore GPU memory
    gpu_image = cp.cuda.runtime.ipcOpenMemHandle(mem_handle)
    image = cp.ndarray((512, 512, 3), dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(gpu_image, 0))

    # Apply Thresholding
    thresholded_image = cp.where(image > 128, 255, 0).astype(cp.uint8)

    # Copy result back to CPU
    final_image = cp.asnumpy(thresholded_image)

    # Save final output
    cv2.imwrite("final_output.jpg", final_image)

    print("Thresholding: Applied, Image saved!")

if __name__ == "__main__":
    threshold_ready = multiprocessing.Event()

    threshold_process = multiprocessing.Process(target=threshold_filter)
    threshold_process.start()
    threshold_process.join()
