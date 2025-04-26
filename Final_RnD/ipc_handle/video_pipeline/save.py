import cv2
import cupy as cp
import multiprocessing
import numpy as np

def save_video(queue_out, output_path="output.mp4", fps=30, frame_size=(1280, 720)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    print("save started")
    # return
    while True:
        item = queue_out.get()
        if item is None:
            break
        # print("saving frame")
        mem_handle, shape = item
        size_in_bytes = np.prod(shape) * cp.uint8().itemsize
        ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle)
        base_mem = cp.cuda.UnownedMemory(ptr, size_in_bytes, owner=None)
        frame_gpu = cp.ndarray(shape, dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(base_mem, 0))
        frame_cpu = cp.asnumpy(frame_gpu)

        if writer is None:
            frame_size = (frame_cpu.shape[1], frame_cpu.shape[0])
            writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        writer.write(frame_cpu)

    # print("save done")
    writer.release()