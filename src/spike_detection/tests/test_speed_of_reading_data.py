"""
Compare the speed of reading 2d arrays from .npy and .h5
"""

# region Imports
import numpy as np
import h5py
from time import perf_counter
from random import randint
import torch
# endregion


# region Constants
# Path to .h5 recording file (old maxwell version)
PATH_H5 = r"E:\KosikLab\Deep-Learning-for-Realtime-Spike-Sorting\data\2950\data.raw.h5"
# Path to .npy recording file
PATH_NPY = "/data/MEAprojects/DLSpikeSorter/2953/data.raw.npy"
# Number of trials
N = 10000
# Length of slice
L = 200
# endregion


# region Helper functions
def get_row():
    return randint(0, n_row-L)


def get_col():
    return randint(0, n_col-L)
# endregion


# region Set up
# h5 = h5py.File(PATH_H5)['sig']
npy = np.load(PATH_NPY, mmap_mode="r")

print("Shapes")
n_row, n_col = npy.shape
# print(f".h5: {h5.shape}")
print(f".npy: {npy.shape}")
print()
# endregion

# # region .h5 tests
# print(".h5 - slice both axes")
# start = perf_counter()
# for _ in range(N):
#     data = h5[(i := get_row()): i+L, (j := get_col()): j+L]
#     tensor = torch.tensor(data.astype("int16"))
#     tensor[:] += 1
# end = perf_counter()
# print(end-start)
# print()
#
# print(".h5 - slice 0th axis")
# start = perf_counter()
# for _ in range(N):
#     data = h5[(i := get_row()): i+L, get_col()]
#     tensor = torch.tensor(data.astype("int16"))
#     tensor[:] += 1
# end = perf_counter()
# print(end-start)
# print()
#
# print(".h5 - slice 1st axis")
# start = perf_counter()
# for _ in range(N):
#     data = h5[get_row(), (j := get_col()): j+L]
#     tensor = torch.tensor(data.astype("int16"))
#     tensor[:] += 1
# end = perf_counter()
# print(end-start)
# print()
# # endregion

# region .npy tests
print(".npy - slice both axes")
start = perf_counter()
for _ in range(N):
    data = npy[(i := get_row()): i+L, (j := get_col()): j+L]
    tensor = torch.tensor(data)  # tensor = torch.from_numpy(data.copy())
    tensor[:] += 1
end = perf_counter()
print(end-start)
print()

print(".npy - slice 0th axis")
start = perf_counter()
for _ in range(N):
    data = npy[(i := get_row()): i+L, get_col()]
    tensor = torch.tensor(data)  # tensor = torch.from_numpy(data.copy())
    tensor[:] += 1
end = perf_counter()
print(end-start)
print()

print(".npy - slice 1st axis")
start = perf_counter()
for _ in range(N):
    data = npy[get_row(), (j := get_col()): j+L]
    tensor = torch.tensor(data)  # tensor = torch.from_numpy(data.copy())
    tensor[:] += 1
end = perf_counter()
print(end-start)
print()
# endregion
