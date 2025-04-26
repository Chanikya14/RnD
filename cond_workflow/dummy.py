import torch
import torchvision
from torchvision.ops import nms
import cv2
import numpy as np


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


# 1. Load the pre-trained YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Check if CUDA (GPU) is available and move model to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 2. Load the input image
image = cv2.imread('dogs.png')

# Convert the BGR image to RGB (since OpenCV loads images in BGR format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize the image to match YOLOv5 input dimensions (640x640)
image_resized = cv2.resize(image_rgb, (640, 640))

# 3. Run object detection
# Convert image to tensor and move it to the GPU
image_tensor = torch.from_numpy(image_resized).float().to(device) / 255.0  # Normalize to [0, 1]
image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Change to (1, 3, H, W)

# Example: Output tensor shape [1, num_boxes, 85] (Batch size of 1)
output = model(image_tensor)  # Shape [1, num_boxes, 85]

# Example dimensions: [1, num_boxes, 85]
# The model output is usually of shape [batch_size, num_boxes, num_classes + 5]

# Extracting values from the output
boxes = output[0][:, :4]  # Bounding box coordinates (x1, y1, x2, y2)
objectness_scores = output[0][:, 4]  # Objectness score
class_scores = output[0][:, 5:]  # Class scores (for each class)

# Get the class with the highest score for each detection
confidence, predicted_classes = class_scores.max(dim=1)

# Apply objectness threshold to filter out low-confidence boxes
confidence_threshold = 0.5  # You can adjust this threshold
objectness_threshold = 0.5
mask = (objectness_scores > objectness_threshold) & (confidence > confidence_threshold)

# Filtered boxes, class predictions, and scores
filtered_boxes = boxes[mask]
filtered_classes = predicted_classes[mask]
filtered_scores = confidence[mask]

# Apply Non-Maximum Suppression (NMS) to remove duplicate boxes
# NMS requires the boxes, scores, and IoU threshold for filtering
iou_threshold = 0.4  # You can adjust this threshold
keep = nms(filtered_boxes, filtered_scores, iou_threshold)

# Keep only the boxes that are selected by NMS
final_boxes = filtered_boxes[keep]
final_classes = filtered_classes[keep]
final_scores = filtered_scores[keep]

# Now you have the final boxes, classes, and scores for your detections
n = len(final_boxes)

for i in range(n):
    print(final_boxes[i])