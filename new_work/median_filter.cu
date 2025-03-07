#include <opencv2/opencv.hpp>
#include <iostream>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

void applyMedianFilter() {
    cv::Mat image = cv::imread("step1.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Step1 image missing!" << std::endl;
        return;
    }

    cv::medianBlur(image, image, 5);
    cv::imwrite("step2.jpg", image);
    std::cout << "Median Filter: Applied, signaling sharpen_filter..." << std::endl;

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
    std::cout << "Median Filter: Waiting for signal..." << std::endl;
    pause(); // Wait for signal, then terminate
}
