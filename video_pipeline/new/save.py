import cv2
import cupy as cp
import multiprocessing

def save_video(queue_out, output_path="output.mp4", fps=30, frame_size=(1280, 720)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while True:
        mem_handle, shape = queue_out.get()
        if mem_handle is None:
            break  # Exit signal

        ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle)
        base_mem = cp.cuda.UnownedMemory(ptr, int(cp.prod(shape)), owner=None)
        frame_gpu = cp.ndarray(shape, dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(base_mem, 0))
        
        frame_cpu = cp.asnumpy(frame_gpu)  # Move to CPU
        writer.write(frame_cpu)

    writer.release()
