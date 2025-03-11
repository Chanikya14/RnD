#include <opencv2/opencv.hpp>
#include <iostream>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "Handle/Cuda_Ipc_Manager.h"

__global__ void thresholdKernel(unsigned char* input, unsigned char* output, int width, int height, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    output[idx] = (input[idx] > threshold) ? 255 : 0;
}

void applyThreshold() {
    cv::Mat image = cv::imread("step3.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Step3 image missing!" << std::endl;
        return;
    }

    int width = image.cols;
    int height = image.rows;
    int size = width * height;

    // Allocate GPU memory
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy image to GPU
    cudaMemcpy(d_input, image.data, size, cudaMemcpyHostToDevice);

    // Launch CUDA Kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    thresholdKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, 128);
    cudaDeviceSynchronize();

    // Copy result back to CPU
    cudaMemcpy(image.data, d_output, size, cudaMemcpyDeviceToHost);
    cv::imwrite("output.jpg", image);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Thresholding (CUDA): Applied, signaling final.cu..." << std::endl;

    // Send signal to next process
    pid_t pid = std::stoi(std::string(std::getenv("FINAL_PID")));
    kill(pid, SIGUSR1);

    exit(0); // Terminate after processing
}

void signalHandler(int signum) {
    if (signum == SIGUSR1) {
        applyThreshold();
    }
}

int main() {
    signal(SIGUSR1, signalHandler);
    std::cout << "Thresholding (CUDA): Waiting for signal..." << std::endl;
    pause(); // Wait for signal, then terminate
}
