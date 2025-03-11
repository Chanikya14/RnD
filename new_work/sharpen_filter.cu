#include <opencv2/opencv.hpp>
#include <iostream>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "Handle/Cuda_Ipc_Manager.h"

__global__ void sharpenKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    int kernel[3][3] = {
        { 0, -1,  0 },
        {-1,  5, -1 },
        { 0, -1,  0 }
    };

    for (int c = 0; c < channels; c++) {
        int sum = 0;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);
                int nidx = (ny * width + nx) * channels + c;
                sum += input[nidx] * kernel[ky + 1][kx + 1];
            }
        }
        output[idx + c] = min(max(sum, 0), 255);
    }
}

void applySharpenFilter() {
    cv::Mat image = cv::imread("step2.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Step2 image missing!" << std::endl;
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
    sharpenKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    cudaDeviceSynchronize();

    // Copy result back to CPU
    cudaMemcpy(image.data, d_output, size, cudaMemcpyDeviceToHost);
    cv::imwrite("step3.jpg", image);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Sharpen Filter (CUDA): Applied, signaling threshold..." << std::endl;

    // Send signal to next process
    pid_t pid = std::stoi(std::string(std::getenv("THRESHOLD_PID")));
    kill(pid, SIGUSR1);

    exit(0); // Terminate after processing
}

void signalHandler(int signum) {
    if (signum == SIGUSR1) {
        applySharpenFilter();
    }
}

int main() {
    signal(SIGUSR1, signalHandler);
    std::cout << "Sharpen Filter (CUDA): Waiting for signal..." << std::endl;
    pause(); // Wait for signal, then terminate
}
