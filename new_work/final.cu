#include <iostream>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

void finalStep() {
    std::cout << "Final Step: Processing completed! Output saved as output.jpg." << std::endl;
    exit(0);
}

void signalHandler(int signum) {
    if (signum == SIGUSR1) {
        finalStep();
    }
}

int main() {
    signal(SIGUSR1, signalHandler);
    std::cout << "Final Step: Waiting for signal..." << std::endl;
    pause(); // Wait for signal, then terminate
}
