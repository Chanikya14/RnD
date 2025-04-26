import torch
import torch.multiprocessing as mp
from capture import capture_video
from process import process_frame
from save import save_results

if __name__ == "__main__":
    # Allocate shared GPU memory
    gpu_buffer = torch.zeros((480, 640, 3), dtype=torch.uint8, device="cuda")

    # Get CUDA IPC handle
    ipc_handle = torch.cuda.memory._ipc_get_handle(gpu_buffer)

    # Create synchronization events
    frame_ready = mp.Event()
    frame_processed = mp.Event()
    frame_processed.set()

    # Start processes
    p1 = mp.Process(target=capture_video, args=("input_video.mp4", ipc_handle, frame_ready, frame_processed))
    p2 = mp.Process(target=process_frame, args=(ipc_handle, frame_ready, frame_processed))
    p3 = mp.Process(target=save_results, args=(ipc_handle, frame_ready, frame_processed))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
