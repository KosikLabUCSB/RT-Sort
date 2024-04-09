"""
Compare speed of reading data from numpy arrays
Load full array into ram vs mmap
"""

import numpy as np
import torch
from time import perf_counter

# Path of .npy array
ARRAY_PATH = "/data/MEAprojects/DLSpikeSorter/2950/data.raw.npy"
# Number of trials
N_TRIALS = 100000
# Length of slice to take from array
LEN_SLICE = 20000


array = np.load(ARRAY_PATH)
start = perf_counter()
for _ in range(N_TRIALS):
    row = np.random.choice(array.shape[0])
    col = np.random.choice(array.shape[1] - LEN_SLICE)
    data = torch.tensor(array[row, col:col + LEN_SLICE], dtype=torch.float32)
    data[LEN_SLICE // 2] += 10
end = perf_counter()
print(f"ram: {end - start}")


array = np.load(ARRAY_PATH, mmap_mode="r")
start = perf_counter()
for _ in range(N_TRIALS):
    row = np.random.choice(array.shape[0])
    col = np.random.choice(array.shape[1]-LEN_SLICE)
    data = torch.tensor(array[row, col:col+LEN_SLICE], dtype=torch.float32)
    data[LEN_SLICE // 2] += 10
end = perf_counter()
print(f"mmap: {end - start}")


