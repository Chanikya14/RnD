#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <cuda_runtime.h>

__global__ void medianFilter(uchar3* input, uchar3* output, int width, int height);
__global__ void sharpenFilter(uchar3* input, uchar3* output, int width, int height);
__global__ void thresholding(uchar3* image, int width, int height, int threshold_value);

#endif
