import cv2
import numpy as np
from multiprocessing import shared_memory
import time

def capture_video(video_path, shm_name, capture_done, process_done, save_done):
    cap = cv2.VideoCapture(video_path)
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video.")
        return

    frame_shape = frame.shape
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=frame.nbytes + 12)  # Frame + shape (3 integers for height, width, channels)

    # Write the shape of the frame in shared memory (12 bytes: 3 * int32)
    np.ndarray(3, dtype=np.int32, buffer=shm.buf, offset=0)[:] = frame_shape
    # shm_names['frame'] = shm.name  # Store shared memory name

    while cap.isOpened():
        # print("cap")
        ret, frame = cap.read()
        if not ret:
            break

        # Write frame to shared memory
        shm.buf[12:] = frame.ravel().tobytes()  # Offset by 12 to skip shape
        capture_done.set()  # Signal processing can start
        save_done.wait()    # Wait till saving is done
        save_done.clear()

    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    cap.release()
    # No need for queue, just a None signal for each process
    # shm_names['frame'] = None
