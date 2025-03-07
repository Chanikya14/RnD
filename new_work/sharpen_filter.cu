#include <opencv2/opencv.hpp>
#include <iostream>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

void applySharpenFilter() {
    cv::Mat image = cv::imread("step2.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Step2 image missing!" << std::endl;
        return;
    }

    // Sharpening kernel
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 
        0, -1,  0,
       -1,  5, -1,
        0, -1,  0);
    cv::filter2D(image, image, image.depth(), kernel);

    cv::imwrite("step3.jpg", image);
    std::cout << "Sharpen Filter: Applied, signaling threshold..." << std::endl;

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
    std::cout << "Sharpen Filter: Waiting for signal..." << std::endl;
    pause(); // Wait for signal, then terminate
}
