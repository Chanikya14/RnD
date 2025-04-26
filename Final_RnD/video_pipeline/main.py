import multiprocessing as mp
from capture import capture_video
from process import process_frames
from save import save_video
import time

from multiprocessing import Queue, Process, Event

def main():
    queue_in = Queue()
    queue_out = Queue()
    process_done_event = Event()
    process_done_event.set()  # allow first frame to be captured

    capture_process = Process(target=capture_video, args=("input_video.mp4", queue_in, process_done_event))
    process_process = Process(target=process_frames, args=(queue_in, queue_out, process_done_event))
    save_process = Process(target=save_video, args=(queue_out,))

    capture_process.start()
    process_process.start()
    save_process.start()

    capture_process.join()
    process_process.join()
    save_process.join()


if __name__ == "__main__":
    main()
