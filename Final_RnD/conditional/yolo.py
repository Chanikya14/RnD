import cupy as cp
import numpy as np
from multiprocessing import shared_memory
import time
import cv2
import torch
import torch.nn.functional as F
from multiprocessing import Process


def check(gpu_image):
    cpu_image = gpu_image.get()  # shape: (H, W, 3), dtype: uint8

    print(cpu_image.shape)
    # Step 3: Display or save using OpenCV
    cv2.imshow('Recovered Image', cpu_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def yolo(input_mem_name):
    start = time.time()
    input_mem = shared_memory.SharedMemory(name=input_mem_name)

    handle_size = 64
    handle = input_mem.buf[:handle_size]
    mem_handle_bytes = bytes(handle)
    img_shape = np.ndarray(3, dtype=np.int32, buffer=input_mem.buf, offset=handle_size)

    print("yolo: Shape received", img_shape)

    cp.cuda.Device(0).use()
    mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle_bytes)
    mem = cp.cuda.memory.MemoryPointer(
        cp.cuda.memory.UnownedMemory(mem_ptr, img_shape.prod() * cp.uint8().itemsize, cp.cuda.Device(0)), 0
    )

    # Convert to CuPy array
    gpu_image = cp.ndarray(tuple(img_shape), dtype=cp.uint8, memptr=mem)
    print("yolo")
    # check(gpu_image)

    # # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cuda:0')

    # Run inference
    results = model(cp.asnumpy(gpu_image))
    # Parse detections
    detections = results.pred[0]  # Tensor of shape (num_detections, 6) → [x1, y1, x2, y2, conf, cls]

    # Get class IDs
    class_ids = detections[:, 5].int().tolist()

    l = []
    # Map to class names
    for cls_id in class_ids:
        l.append(model.names[int(cls_id)])



    # results.save()
    cp.cuda.runtime.ipcCloseMemHandle(mem_ptr)


    shm = shared_memory.SharedMemory(name="flag", create=True, size=1)
    flag = np.ndarray((1,), dtype=np.uint8, buffer=shm.buf)
    flag[0] = 00 

    end = time.time()
    elapsed_time = end - start
    print(f"yolo Elapsed time: {elapsed_time:.2f} seconds")

    if "cat" in l:
        print("CAT detected!!")
        flag[0] |= 0b01 
        
    elif "dog" in l:
        print("DOG detected!!")
        flag[0] |= 0b10 
        
    else:
        print("Neither cat nor dog detected")


    print("====================END===================")

    time.sleep(20)
    input_mem.close()

if __name__ == "__main__":
    input_mem_name = "handle1"
    yolo(input_mem_name)