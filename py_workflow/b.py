import numpy as np
from multiprocessing import shared_memory

# Attach to existing shared memory
shared_mem = shared_memory.SharedMemory(name="my_shared_mem")

# Read data from shared memory
np_array = np.ndarray((10,), dtype=np.int32, buffer=shared_mem.buf)
print("Reader: Data read from shared memory:", np_array)

# Cleanup
shared_mem.close()
