import cv2
import cupy as cp
import numpy as np
from multiprocessing import shared_memory

def draw_boxes(shared_mem_name):
    shared_mem = shared_memory.SharedMemory(name=shared_mem_name)
    mem_handle_bytes = np.frombuffer(shared_mem.buf, dtype=np.uint8)

    mem_handle = cp.cuda.runtime.ipcGetMemHandle(mem_handle_bytes.ctypes.data)
    gpu_image = cp.asarray(mem_handle, dtype=cp.uint8)

    frame = cp.asnumpy(gpu_image)  # Convert to NumPy for OpenCV processing
    
    for _, row in detect_objects(shared_mem_name).iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label, conf = row['name'], row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    shared_mem.close()
    