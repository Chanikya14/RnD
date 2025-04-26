import cv2
import cupy as cp
import multiprocessing as mp
from cupy.cuda.runtime import ipcGetMemHandle
import time

def capture_video(video_path, queue, event):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        event.wait()
        event.clear()
        ret, frame = cap.read()
        if not ret:
            break

        frame_gpu = cp.asarray(frame, dtype=cp.uint8)  # Move frame to GPU
        mem_handle = ipcGetMemHandle(frame_gpu.data.ptr)  # Get IPC handle

        print("main queue in")
        queue.put((mem_handle, frame_gpu.shape))  # Send to processing queue
        # time.sleep(10)

    cap.release()
    queue.put(None)  # Signal end of video
