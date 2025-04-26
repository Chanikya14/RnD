import cv2
import cupy as cp
import torch
import numpy as np
import torch.multiprocessing as mp
from multiprocessing import shared_memory
import time


def capture_video(video_path, shared_mem_name, frame_ready, frame_processed):
    cap = cv2.VideoCapture(video_path)
    handle_size = 64
    shared_mem = shared_memory.SharedMemory(name=shared_mem_name, create=True, size=handle_size + 12)
    # shared_mem = 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_processed.wait()  # Wait for previous frame to be used

        # Copy frame to GPU
        gpu_image = cp.asarray(frame, dtype=cp.uint8)

        # Get CUDA IPC memory handle
        mem_handle = cp.cuda.runtime.ipcGetMemHandle(gpu_image.data.ptr)
        mem_handle_bytes = np.frombuffer(mem_handle, dtype=np.uint8)

        # Ensure correct handle size
        handle_size = len(mem_handle)

        # Store handle
        shared_mem.buf[:handle_size] = mem_handle_bytes

        # Store shape
        np.ndarray(3, dtype=np.int32, buffer=shared_mem.buf, offset=handle_size)[:] = gpu_image.shape

        frame_ready.set()
        frame_processed.clear()
        # time.sleep(5)
        return

    cap.release()

    shared_mem.close()
    shared_mem.unlink()

if __name__ == "__main__":
    video_path = "input_video.mp4"
    shared_mem_name = "capture"

    # Create synchronization events
    frame_ready = mp.Event()
    frame_processed = mp.Event()
    frame_processed.set()

    # Start capture process
    p = mp.Process(target=capture_video, args=(video_path, shared_mem_name, frame_ready, frame_processed))
    p.start()
