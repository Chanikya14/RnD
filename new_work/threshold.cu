#include <opencv2/opencv.hpp>
#include <iostream>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

void applyThreshold() {
    cv::Mat image = cv::imread("step3.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Step3 image missing!" << std::endl;
        return;
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, gray, 128, 255, cv::THRESH_BINARY);

    cv::imwrite("output.jpg", gray);
    std::cout << "Thresholding: Applied, signaling final.cu..." << std::endl;

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
    std::cout << "Thresholding: Waiting for signal..." << std::endl;
    pause(); // Wait for signal, then terminate
}
