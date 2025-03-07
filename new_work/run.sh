# # nvcc -o image_pipeline main.cu grayscale.cu median_filter.cu threshold.cu -lopencv_core -lopencv_imgcodecs -lcudart
# nvcc -o image_pipeline main.cu median_filter.cu sharpen_filter.cu threshold.cu `pkg-config --cflags --libs opencv4`

# ./image_pipeline

#!/bin/bash

# Start processes and capture PIDs
./final & echo $! > final.pid
export FINAL_PID=$(cat final.pid)

./threshold & echo $! > threshold.pid
export THRESHOLD_PID=$(cat threshold.pid)

./sharpen_filter & echo $! > sharpen.pid
export SHARPEN_PID=$(cat sharpen.pid)

./median_filter & echo $! > median.pid
export MEDIAN_PID=$(cat median.pid)

# Debugging output
echo "Exported PIDs:"
echo "MEDIAN_PID=$MEDIAN_PID"
echo "SHARPEN_PID=$SHARPEN_PID"
echo "THRESHOLD_PID=$THRESHOLD_PID"
echo "FINAL_PID=$FINAL_PID"

# Run main in the same shell session with variables
./main
