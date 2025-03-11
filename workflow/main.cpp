#include <opencv2/opencv.hpp>
#include <iostream>
#include "cuda_ipc_manager.h"

extern void captureFrame(cv::VideoCapture &cap, CudaIpcManager &ipcManager, size_t size);
extern void preprocessFrame(CudaIpcManager &ipcManager, int width, int height);
extern void detectObjects(CudaIpcManager &ipcManager, int width, int height);
extern void displayFrame(CudaIpcManager &ipcManager, int width, int height);

int main() {
    cv::VideoCapture cap("input_video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    size_t frameSize = width * height * 3;

    CudaIpcManager ipcManager("meta_shm", "data_shm");

    while (true) {
        captureFrame(cap, ipcManager, frameSize);
        preprocessFrame(ipcManager, width, height);
        detectObjects(ipcManager, width, height);
        displayFrame(ipcManager, width, height);

        if (cv::waitKey(30) == 'q') break;
    }

    cap.release();
    return 0;
}
