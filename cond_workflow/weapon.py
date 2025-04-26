import torch
import cv2
import numpy as np

# 1. Load the pre-trained YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 2. Load the input image
image = cv2.imread('dogs.png')

# Convert the BGR image to RGB (since OpenCV loads images in BGR format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Run object detection
results = model(image)

print(results.shape)

# 4. Parse the results
# The results are in a list of dictionaries, but we can also use the results directly for convenience
labels = results.names  # List of class labels
predictions = results.xyxy[0]  # Bounding boxes [x1, y1, x2, y2, confidence, class]

print(len(predictions))
# 5. Action based on object detection
weapon_detected = False
for *box, conf, cls in predictions:
    label = labels[int(cls)]
    print(label)
    if label == "dog" and conf > 0.5:  # You can adjust the confidence threshold
        weapon_detected = True
    # Draw a bounding box on the image
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
