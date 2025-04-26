import multiprocessing as mp
from capture import capture_video
from process import process_frames
from save import save_video
import time
from multiprocessing import Event

def main():
    # Initialize shared memory names here
    shm_names = "shm"

    capture_done = Event()
    process_done = Event()
    save_done = Event()

    # Start processes
    capture_process = mp.Process(target=capture_video, args=("input_video.mp4", shm_names, capture_done, process_done, save_done))
    process_process = mp.Process(target=process_frames, args=(shm_names, capture_done, process_done, save_done))
    save_process = mp.Process(target=save_video, args=(shm_names,capture_done, process_done, save_done))

    capture_process.start()
    process_process.start()
    save_process.start()

    capture_process.join()
    process_process.join()
    save_process.join()


if __name__ == "__main__":
    main()
