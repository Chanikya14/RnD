import cupy as cp
import multiprocessing
from multiprocessing.shared_memory import SharedMemory

def sharpen_filter(shared_mem_name):
    print("Sharpen Filter: Waiting for signal...")
    
    # Open shared memory to get memory handle from median filter
    shared_mem = SharedMemory(name=shared_mem_name)
    mem_handle = bytes(shared_mem.buf[:])  # Read memory handle from shared memory
    shared_mem.close()

    # Open CUDA memory using IPC
    mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle)
    if mem_ptr is None or mem_ptr == 0:
        raise RuntimeError("Failed to open memory handle!")

    # Map the memory to CuPy
    mem = cp.cuda.memory.BaseMemory(mem_ptr)
    gpu_image = cp.ndarray((512, 512, 3), dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(mem, 0))

    # Sharpening kernel
    sharpen_kernel = cp.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=cp.float32)

    # Apply convolution (sharpening)
    sharpened_image = cp.apply_along_axis(lambda x: cp.convolve(x, sharpen_kernel, mode='same'), 2, gpu_image)

    # Allocate new GPU memory for sharpened image
    sharpened_gpu_image = cp.array(sharpened_image, copy=True)

    # Create shared memory for the sharpened image's CUDA memory handle
    new_mem_handle = cp.cuda.runtime.ipcGetMemHandle(sharpened_gpu_image.data.ptr)
    new_shared_mem = SharedMemory(name="sharpened_ipc_handle", create=True, size=len(new_mem_handle))
    new_shared_mem.buf[:len(new_mem_handle)] = new_mem_handle
    new_shared_mem.close()

    print("Sharpen Filter: Applied, signaling thresholding...")
    threshold_ready.set()  # Signal the next process

if __name__ == "__main__":
    shared_mem_name = "filtered_ipc_handle"
    threshold_ready = multiprocessing.Event()

    sharpen_process = multiprocessing.Process(target=sharpen_filter, args=(shared_mem_name,))
    sharpen_process.start()
    sharpen_process.join()
