import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

SMALL_WINDOW_SIZE = 10
LARGE_WINDOW_SIZE = 21

output = np.load(r"E:\KosikLab\Deep-Learning-for-Realtime-Spike-Sorting\data\outputs.npy")
loc_scores = output[1:-1]
loc_probs = np.exp(loc_scores) / np.sum(np.exp(loc_scores))  # softmax
# loc_probs *= 100

n1_max = 0
i_max = 0
n1s = []
for i in range(loc_probs.size - SMALL_WINDOW_SIZE + 1):
    n1 = np.sum(loc_probs[i: i+SMALL_WINDOW_SIZE])
    n2 = np.sum(loc_probs[(c := i+SMALL_WINDOW_SIZE // 2) - LARGE_WINDOW_SIZE // 2: c + LARGE_WINDOW_SIZE // 2])
    if n1 > n1_max:
        n1_max = n1
        i_max = i
    # n2 += 1e-9
    n1s.append(n1 / n2)
x = np.arange(loc_probs.size - SMALL_WINDOW_SIZE + 1) + SMALL_WINDOW_SIZE // 2
plt.plot(loc_scores, color="black")
# plt.plot(x, n1s)
plt.show()

