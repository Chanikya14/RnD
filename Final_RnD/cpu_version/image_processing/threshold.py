import cv2
import numpy as np

def threshold_filter(input_image_path, output_image_path, threshold_value=128):
    print("VPU Threshold Filter: Loading sharpened image...")

    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error: Cannot load image!")
    
    print(f"VPU Threshold Filter: Image loaded with shape {image.shape}")

    print(f"VPU Threshold Filter: Applying threshold at value {threshold_value}...")
    _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    print(f"VPU Threshold Filter: Saving thresholded image to {output_image_path}...")
    cv2.imwrite(output_image_path, thresholded)

if __name__ == "__main__":
    input_image_path = "sharpen_output_vpu.png"
    output_image_path = "final_output_vpu.png"
    threshold_filter(input_image_path, output_image_path)
