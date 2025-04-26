import torch
import cupy as cp
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from multiprocessing import shared_memory

class ObjectDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load YOLOv5 model
    
    def detect(self, frame):
        results = self.model(frame)  # Run detection
        return results.pandas().xyxy[0]  # Convert detections to a Pandas DataFrame

def process_frame(in_shared_mem, frame_ready, frame_processed, out_shared_mem):

    detector = ObjectDetector()  # Initialize YOLOv5 detector

    while True:
        frame_ready.wait()  # Wait for new frame

            # Open shared memory for input
        input_mem = shared_memory.SharedMemory(name=in_shared_mem)

        # Read memory handle
        handle_size = 64  # Ensure correct IPC handle size
        mem_handle_bytes = bytes(input_mem.buf[:handle_size])

        # Read image shape
        img_shape = np.ndarray(3, dtype=np.int32, buffer=input_mem.buf, offset=handle_size)

        print("Median Filter: Shape received", img_shape)

        # Open CUDA memory using IPC
        cp.cuda.Device(0).use()  # Ensure CUDA is initialized
        mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle_bytes)

        # Create CuPy MemoryPointer
        mem = cp.cuda.memory.MemoryPointer(
            cp.cuda.memory.UnownedMemory(mem_ptr, img_shape.prod() * cp.uint8().itemsize, cp.cuda.Device(0)), 0
        )

        # Convert to CuPy array
        frame = cp.ndarray(tuple(img_shape), dtype=cp.uint8, memptr=mem)

        # Perform object detection
        detections = detector.detect(frame)

        # Send detection results to the next process
        result_queue.put(detections.to_dict())

        print("Frame processed on GPU with YOLOv5")

        frame_processed.set()
        frame_ready.clear()

if __name__ == "__main__":
    in_shared_mem = "capture"
    out_shared_mem = "save"

    frame_ready = mp.Event()
    frame_processed = mp.Event()
    frame_processed.set()

    result_queue = mp.Queue()  # Queue to send detections

    p = mp.Process(target=process_frame, args=(in_shared_mem, frame_ready, frame_processed, out_shared_mem))
    p.start()
