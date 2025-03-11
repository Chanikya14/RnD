nvcc -o main main.cu Handle/Cuda_Ipc_Manager.cu `pkg-config --cflags --libs opencv4` --diag-suppress=611
nvcc -o median_filter median_filter.cu Handle/Cuda_Ipc_Manager.cu `pkg-config --cflags --libs opencv4` --diag-suppress=611
nvcc -o sharpen_filter sharpen_filter.cu Handle/Cuda_Ipc_Manager.cu `pkg-config --cflags --libs opencv4` --diag-suppress=611
nvcc -o threshold threshold.cu Handle/Cuda_Ipc_Manager.cu `pkg-config --cflags --libs opencv4` --diag-suppress=611
nvcc -o final final.cu Handle/Cuda_Ipc_Manager.cu `pkg-config --cflags --libs opencv4` --diag-suppress=611
