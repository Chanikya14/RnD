import cv2
import numpy as np
import torch
from multiprocessing import shared_memory
import time

def process_frames(shm_name, capture_done, process_done, save_done):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cuda:0')  # Use 'cpu' instead of 'cuda:0'
    names = model.names  # class index to label name

    while True:
        capture_done.wait()  # Wait for capture to complete
        capture_done.clear()
        # print("yolo")
        # Open shared memory and get frame pointer
        shm = shared_memory.SharedMemory(name=shm_name)
        shape = np.ndarray((3,), dtype=np.int32, buffer=shm.buf[:12])

        # Now map the actual frame
        frame_cpu = np.ndarray(shape=tuple(shape), dtype=np.uint8, buffer=shm.buf[12:])

        # Run inference (on CPU)
        frame_float32 = frame_cpu.astype(np.float32) / 255.0
        results = model(frame_cpu)
     
        # Parse detections and annotate the frame
        detections = results.pred[0]  # Tensor of shape (num_detections, 6)
        # print(len(detections))
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f'{names[int(cls)]} {conf:.2f}'
            
            # Draw rectangle and label
            cv2.rectangle(frame_cpu, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(frame_cpu, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        shm.buf[12:] = frame_cpu.ravel().tobytes()
        process_done.set() 
        # time.sleep(0.05)  # Simulate delay in processing

    print("Processing finished.")
