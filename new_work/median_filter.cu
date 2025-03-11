#include <opencv2/opencv.hpp>
#include <iostream>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "Handle/Cuda_Ipc_Manager.h"

#define KERNEL_SIZE 5  // 5x5 median filter

__device__ unsigned char getMedian(unsigned char* window, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (window[j] > window[j + 1]) {
                unsigned char temp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = temp;
            }
        }
    }
    return window[size / 2];
}

__global__ void medianFilterKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_k = KERNEL_SIZE / 2;
    int idx = (y * width + x) * channels;

    for (int c = 0; c < channels; c++) {
        unsigned char window[KERNEL_SIZE * KERNEL_SIZE];
        int count = 0;

        for (int dy = -half_k; dy <= half_k; dy++) {
            for (int dx = -half_k; dx <= half_k; dx++) {
                int nx = min(max(x + dx, 0), width - 1);
                int ny = min(max(y + dy, 0), height - 1);
                window[count++] = input[(ny * width + nx) * channels + c];
            }
        }
        output[idx + c] = getMedian(window, KERNEL_SIZE * KERNEL_SIZE);
    }
}

void applyMedianFilter() {
    cv::Mat image = cv::imread("step1.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Step1 image missing!" << std::endl;
        return;
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    int size = width * height * channels;

    // Allocate GPU memory
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy image to GPU
    cudaMemcpy(d_input, image.data, size, cudaMemcpyHostToDevice);

    // Launch CUDA Kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    medianFilterKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    cudaDeviceSynchronize();

    // Copy result back to CPU
    cudaMemcpy(image.data, d_output, size, cudaMemcpyDeviceToHost);
    cv::imwrite("step2.jpg", image);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Median Filter (CUDA): Applied, signaling sharpen_filter..." << std::endl;

    // Send signal to next process
    const char* pid_env = std::getenv("SHARPEN_PID");
    if (!pid_env) {
        std::cerr << "Error: SHARPEN_PID environment variable not set!" << std::endl;
        exit(1);
    }

    pid_t pid = std::stoi(std::string(pid_env));
    kill(pid, SIGUSR1);

    exit(0); // Terminate after processing
}

void signalHandler(int signum) {
    if (signum == SIGUSR1) {
        applyMedianFilter();
    }
}

int main() {
    signal(SIGUSR1, signalHandler);
    std::cout << "Median Filter (CUDA): Waiting for signal..." << std::endl;
    pause(); // Wait for signal, then terminate
}
