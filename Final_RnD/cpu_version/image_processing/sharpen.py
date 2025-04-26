import cv2
import numpy as np

def sharpen_filter(input_image_path, output_image_path):
    print("VPU Sharpen Filter: Loading median-filtered image...")

    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error: Cannot load image!")
    
    print(f"VPU Sharpen Filter: Image loaded with shape {image.shape}")

    print("VPU Sharpen Filter: Applying sharpening kernel...")
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)

    sharpened = cv2.filter2D(image, ddepth=-1, kernel=sharpen_kernel)

    print(f"VPU Sharpen Filter: Saving sharpened image to {output_image_path}...")
    cv2.imwrite(output_image_path, sharpened)

if __name__ == "__main__":
    input_image_path = "median_output_vpu.png"
    output_image_path = "sharpen_output_vpu.png"
    sharpen_filter(input_image_path, output_image_path)
