import torch
import torch.nn.functional as F
import cupy as cp
import pandas as pd

class ObjectDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to('cuda')

    def detect(self, frame_gpu):
        frame_torch = torch.as_tensor(frame_gpu, device="cuda").permute(2, 0, 1).unsqueeze(0) / 255.0
        frame_torch = F.interpolate(frame_torch, size=(frame_torch.shape[2] // 32 * 32, frame_torch.shape[3] // 32 * 32), mode="bilinear")
        return self.model(frame_torch)

def process_frames(queue_in, queue_out):
    detector = ObjectDetector()
    
    while True:
        mem_handle, shape = queue_in.get()
        if mem_handle is None:
            break  # Exit signal

        ptr = cp.cuda.runtime.ipcOpenMemHandle(mem_handle)
        base_mem = cp.cuda.UnownedMemory(ptr, int(cp.prod(shape)), owner=None)
        frame_gpu = cp.ndarray(shape, dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(base_mem, 0))
        
        frame_copy_gpu = cp.copy(frame_gpu)
        detections = detector.detect(frame_copy_gpu)[0]

        conf_threshold = 0.25
        valid_detections = detections[detections[:, 4] > conf_threshold]
        boxes = valid_detections[:, :4].cpu().numpy().astype(int)
        scores = valid_detections[:, 4].cpu().numpy()
        labels = valid_detections[:, 5].cpu().numpy()

        detections_df = pd.DataFrame({'xmin': boxes[:, 0], 'ymin': boxes[:, 1], 'xmax': boxes[:, 2], 'ymax': boxes[:, 3], 'confidence': scores, 'name': labels})

        cp.cuda.runtime.ipcCloseMemHandle(ptr)

        # Draw bounding boxes
        for _, row in detections_df.iterrows():
            x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            frame_copy_gpu[y1:y2, x1:x2] = cp.array([255, 255, 0], dtype=cp.uint8)

        new_handle = cp.cuda.runtime.ipcGetMemHandle(frame_copy_gpu.data.ptr)
        queue_out.put((new_handle, frame_copy_gpu.shape))
