from median import median_filter
from sharpen import sharpen_filter
from threshold import threshold_filter
import time

def run_vpu_pipeline():
    t1 = time.time()
    median_filter("nebula.jpeg", "median_output_vpu.png")
    t2 = time.time()
    print(f"median Elapsed: {t2 - t1:.4f} seconds")
    sharpen_filter("median_output_vpu.png", "sharpen_output_vpu.png")
    t3 = time.time()
    print(f"sharpen Elapsed: {t3 - t2:.4f} seconds")
    threshold_filter("sharpen_output_vpu.png", "final_output_vpu.png")
    t4 = time.time()
    print(f"threshold Elapsed: {t4 - t3:.4f} seconds")

    print(f"Total Elapsed: {t4 - t1:.4f} seconds")

if __name__ == "__main__":
    run_vpu_pipeline()
