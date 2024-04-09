import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from scipy.stats import norm
from scipy.signal import find_peaks

LOC_THRESH = 10

x = np.arange(150)
# y = norm.pdf(x, loc=60, scale=1)
# y += norm.pdf(x, loc=120, scale=0.5)
# y = (y + 0.5) * 100
y = np.zeros_like(x)
y[0] = 100
peaks = find_peaks(y, height=LOC_THRESH)
print(peaks)


_, ax = plt.subplots(1)  # type: None, Axes
ax.plot(x, y)
ax.axhline(LOC_THRESH, color="black", linestyle="dashed")
ax.set_ylim(0, 100)
plt.show()
