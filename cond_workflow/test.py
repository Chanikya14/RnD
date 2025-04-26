import cv2
import torch

# Load YOLOv5 model with GPU
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cuda:0')  # or 'cpu' if needed

# Read image using OpenCV
img = cv2.imread('cat.jpeg')

a, b = img.shape[1], img.shape[0]

s = 0.8
t = 10

# Resize to desired size (e.g., 640x640)
img_resized = cv2.resize(img, (int(a*s), int(b*s)+t))

# Convert BGR (OpenCV format) to RGB (YOLO expects RGB)
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# Run YOLOv5 inference (can pass numpy array directly)
results = model(img_rgb)

# Print and visualize results
results.print()
results.show()  # Opens image with bounding boxes
# results.save()  # Saves result images to 'runs/detect/exp'
