import cv2
import numpy as np

def median_filter(input_image_path, output_image_path):
    print("VPU Median Filter: Loading image...")
    
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error: Cannot load image!")
    
    print(f"VPU Median Filter: Image loaded with shape {image.shape}")

    print("VPU Median Filter: Applying 5x5 median filter...")
    median_filtered = cv2.medianBlur(image, ksize=5)

    print(f"VPU Median Filter: Saving median-filtered image to {output_image_path}...")
    cv2.imwrite(output_image_path, median_filtered)

if __name__ == "__main__":
    input_image_path = "nebula.jpeg"
    output_image_path = "median_output_vpu.png"
    median_filter(input_image_path, output_image_path)
