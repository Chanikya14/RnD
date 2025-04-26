import multiprocessing as mp
from capture import capture_video
from process import process_frames
from save import save_video

def main():
    queue_in = mp.Queue()
    queue_out = mp.Queue()

    capture_process = mp.Process(target=capture_video, args=("input_video.mp4", queue_in))
    process_process = mp.Process(target=process_frames, args=(queue_in, queue_out))
    save_process = mp.Process(target=save_video, args=(queue_out,))

    capture_process.start()
    process_process.start()
    save_process.start()

    capture_process.join()
    process_process.join()
    save_process.join()

if __name__ == "__main__":
    main()
