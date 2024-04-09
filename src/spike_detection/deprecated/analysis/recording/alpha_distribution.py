"""
Visualize the distribution of waveform alphas
"""

import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import numpy as np
from src.data import full_dataset

dataset = full_dataset(20, 0, 0)
alphas = []
for idx in range(len(dataset)):
    _, label = dataset[idx]
    if label[0]:
        alphas.append(label[2].item())

combos = set()
diffs = []
for i in range(len(alphas)):
    for j in range(len(alphas)):
        if j == i:
            continue

        key = tuple(sorted((i, j)))
        if key not in combos:
            combos.add(key)
            diffs.append(alphas[i] - alphas[j])
print(len(diffs))
print(np.max(diffs))
print(np.min(diffs))

print()
print(np.mean(diffs))
print(np.std(diffs))

print()
print(np.median(diffs))
from scipy.stats import iqr
print(iqr(diffs))
