import torch
import torch.nn.functional as F
import cupy as cp
import pandas as pd
import cv2
import numpy as np
import time

def process_frames(queue_in, queue_out, event):
    # detector = ObjectDetector()
    # print("process started")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cuda:0')
    cnt = 0
    while True:
        # print(cnt)
        # cnt += 1
        item = queue_in.get()
        if item is None:
            break  # Exit signal

        mem_handle, shape = item

        ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle)
        base_mem = cp.cuda.UnownedMemory(ptr, np.prod(shape) * cp.uint8().itemsize, owner=None)
        frame_gpu = cp.ndarray(shape, dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(base_mem, 0))
        
        # Run inference
        results = model(cp.asnumpy(frame_gpu))

        # Parse detections
        detections = results.pred[0]  # Tensor of shape (num_detections, 6) → [x1, y1, x2, y2, conf, cls]

        frame_cpu = cp.asnumpy(frame_gpu).copy()

        event.set()
        names = model.names  # class index to label name

        # Draw boxes and labels
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f'{names[int(cls)]} {conf:.2f}'
            
            # Draw rectangle
            cv2.rectangle(frame_cpu, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            
            # Put label
            cv2.putText(frame_cpu, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)


        # Convert back to CuPy
        frame_gpu_annotated = cp.asarray(frame_cpu)

        new_handle = cp.cuda.runtime.ipcGetMemHandle(frame_gpu_annotated.data.ptr)
        # print("putting in out queue")
        queue_out.put((new_handle, frame_gpu_annotated.shape))
    # print("process done")
    queue_out.put(None)
