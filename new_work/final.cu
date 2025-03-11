#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "Handle/Cuda_Ipc_Manager.h"

#define META_SHM_FINAL "final_meta"
#define DATA_SHM_FINAL "final_data"

void finalStep() {
    CudaIpcManager ipc_final(META_SHM_FINAL, DATA_SHM_FINAL);

    // Import GPU memory from final step
    unsigned char* d_output = (unsigned char*)ipc_final.importMemory(READ);
    if (!d_output) {
        std::cerr << "Error: Failed to import GPU memory!" << std::endl;
        return;
    }

    // Set image dimensions
    int width = 675;
    int height = 225;
    int size = width * height;

    // Allocate CPU memory
    std::vector<unsigned char> h_output(size);

    // Copy GPU memory to CPU
    cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);

    // Convert to OpenCV Mat and save
    cv::Mat image(height, width, CV_8UC1, h_output.data());
    cv::imwrite("final.jpeg", image);

    std::cout << "Final Step: Output saved as final.jpeg." << std::endl;
}

int main() {
    std::cout << "Final Step: Processing..." << std::endl;
    finalStep();
    return 0;
}
