import cv2
import torch
import numpy as np
import pandas as pd
import torch.multiprocessing as mp

def draw_boxes(frame, detections):
    for index, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf, label = row['confidence'], row['name']
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return frame

def save_results(ipc_handle, frame_ready, frame_processed, result_queue):
    # Convert IPC handle back to a CUDA tensor
    gpu_buffer = torch.cuda.ByteTensor((480, 640, 3))
    gpu_buffer = torch.cuda.memory._ipc_open_handle(ipc_handle)

    while True:
        frame_ready.wait()  # Wait for processed frame

        # Copy frame from GPU to CPU
        processed_frame = gpu_buffer.cpu().numpy()

        # Retrieve detection results
        detections = result_queue.get()
        detections = pd.DataFrame(detections)  # Convert back to DataFrame

        # Draw bounding boxes
        processed_frame = draw_boxes(processed_frame, detections)

        # Display or save frame
        cv2.imshow("YOLOv5 Processed Frame", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_processed.set()
        frame_ready.clear()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    ipc_handle = None  

    frame_ready = mp.Event()
    frame_processed = mp.Event()
    frame_processed.set()

    result_queue = mp.Queue()

    p = mp.Process(target=save_results, args=(ipc_handle, frame_ready, frame_processed, result_queue))
    p.start()
