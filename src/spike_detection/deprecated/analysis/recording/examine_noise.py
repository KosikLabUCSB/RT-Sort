"""
Look at variation of noise throughout recordings

Conclusions:

Within small time windows, std of noise = ~1
Noise (mean of all samples not in spike region) varies across different channels
"""

import numpy as np
import matplotlib.pyplot as plt

raw = np.load(r"E:\KosikLab\Deep-Learning-for-Realtime-Spike-Sorting\data\2953\data.raw.npy", mmap_mode="r")
spike_times = np.load(r"E:\KosikLab\Deep-Learning-for-Realtime-Spike-Sorting\data\2953\sorted.npz", allow_pickle=True)["spike_times"]

SPIKE_BUFFER = 100
failed_times = [t for st in spike_times for t in range(st - SPIKE_BUFFER, st + SPIKE_BUFFER + 1)]
sample_times = np.setdiff1d(np.arange(raw.shape[1]), failed_times)

CHANNEL = 231
WINDOW_SIZE = 100000000000
noise = raw[CHANNEL, :]
print(np.std(noise))
exit()
plt.hist(noise, bins=50)
plt.show()

# plt.plot(noise)
# plt.show()
#
# mean_total = 0
# std_total = 0
# window_ind = range(0, noise.size, WINDOW_SIZE)
# for i in window_ind:
#     window = noise[i:i+WINDOW_SIZE]
#     mean = np.mean(window)
#     std = np.std(window)
#     mean_total += mean
#     std_total += std
#     print(f"Mean: {mean:.2f} | STD: {std:.2f}")
# print(mean_total / len(window_ind))
# print(std_total / len(window_ind))


