#include <opencv2/opencv.hpp>
#include <iostream>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include "Handle/Cuda_Ipc_Manager.h"

int main() {
    cv::Mat image = cv::imread("nebula.jpeg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    cv::imwrite("step1.jpg", image); // Save for the next step
    std::cout << "Main: Image loaded, sending signal to median_filter..." << std::endl;

    // Get process ID of median_filter (assumes it's running)
    const char* pid_env = std::getenv("MEDIAN_PID");
    if (!pid_env) {
        std::cerr << "Error: MEDIAN_PID environment variable not set!" << std::endl;
        return 1;
    }

    pid_t pid = std::stoi(std::string(pid_env));
    kill(pid, SIGUSR1); // Send signal to start median filtering

    return 0;
}
