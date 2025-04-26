import cv2
import numpy as np
from multiprocessing import shared_memory

def save_video(shm_name, capture_done, process_done, save_done):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    frame_shape = (1080, 1920, 3)  # Assuming 1080p frames

    while True:
        process_done.wait()  # Wait for processing to complete
        process_done.clear()
        # print("save")
        # Open shared memory and get frame pointer
        shm = shared_memory.SharedMemory(name=shm_name)
        frame_shape = np.ndarray((3,), dtype=np.int32, buffer=shm.buf[:12])
        frame_cpu = np.ndarray(shape=frame_shape, dtype=np.uint8, buffer=shm.buf[12:])  # Assuming 1080p frames

        if writer is None:
            frame_size = (frame_cpu.shape[1], frame_cpu.shape[0])
            writer = cv2.VideoWriter("output_video.mp4", fourcc, 30, frame_size)

        writer.write(frame_cpu)
        save_done.set()
        print("Saving frame...")

    print("Video saving completed.")
    writer.release()
