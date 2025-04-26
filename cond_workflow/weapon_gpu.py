import torch
import cv2
import numpy as np

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

print(COCO_CLASSES[15])

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

# Run inference
results = model(image_tensor)

print(results.shape)

# 4. Parse the results (this change accesses predictions correctly for GPU mode)
labels = results.names  # List of class labels

# Access the predictions (use `.pred[0]` to get the results from GPU)
predictions = results.pred[0]  # Bounding boxes [x1, y1, x2, y2, confidence, class]

print(len(predictions))

# 5. Action based on object detection
weapon_detected = False
for *box, conf, cls in predictions:
    label = labels[int(cls)]
    print(label)
    if label == "dog" and conf > 0.5:  # You can adjust the confidence threshold
        weapon_detected = True
    # Draw a bounding box on the image (move box to CPU if necessary)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
    cv2.putText(image, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 6. Take action based on weapon detection
if weapon_detected:
    print("dog detected! Taking action...")
    # You can add actions here, like sending an alert or logging the event.
else:
    print("No weapon detected.")

# 7. Show the image (optional)
# cv2.imshow("Detected Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Optionally, save the image with detected bounding boxes
cv2.imwrite('output_image.jpg', image)
