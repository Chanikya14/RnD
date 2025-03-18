import numpy as np
from multiprocessing import shared_memory

# Create shared memory
shared_mem = shared_memory.SharedMemory(name="my_shared_mem", create=True, size=1024)

# Write data (an array of 10 integers)
data = np.arange(10, dtype=np.int32)
np_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shared_mem.buf)
np_array[:] = data[:]  # Copy data to shared memory

print("Writer: Data written to shared memory:", np_array)

# Keep running to let reader access
input("Press Enter to cleanup...")

# Cleanup shared memory
shared_mem.close()
shared_mem.unlink()
