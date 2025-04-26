import torch
import cv2
import numpy as np

# ------------------- 1. Capture & Decode Video -------------------
def capture_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

# ------------------- 2. Object Detection with YOLOv5 -------------------
class ObjectDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load YOLOv5 model
    
    def detect(self, frame):
        results = self.model(frame)  # Run detection
        print(results)
        print("=============")
        return results.pandas().xyxy[0]  # Convert detections to a Pandas DataFrame

# ------------------- 3. Draw Bounding Boxes -------------------
def draw_boxes(frame, detections):
    # print(frame.type)
    for index, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf, label = row['confidence'], row['name']
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return frame

# ------------------- 4. Video Processing Pipeline -------------------
def main(video_path):
    detector = ObjectDetector()
    frames = []

    for frame in capture_video(video_path):
        detections = detector.detect(frame)
        frame = draw_boxes(frame, detections)
        frames.append(frame)
    save_video(frames)

# ------------------- 5. Save Video -------------------
def save_video(frames, output_path="output.mp4", fps=60):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

# ------------------- Run Pipeline -------------------
if __name__ == "__main__":
    main("input_video.mp4")
