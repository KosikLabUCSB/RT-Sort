"""
Test creating a multicolored line from a sample trace

https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

y = np.load("sample.npy")
x = np.linspace(0, y.size, 5000)
y = np.interp(x, np.arange(y.size), y)


points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, axs = plt.subplots(1)

lc = LineCollection(segments, cmap='viridis')
lc.set_array(y)
lc.set_linewidth(2)
line = axs.add_collection(lc)
fig.colorbar(line, ax=axs)

axs.set_xlim(x.min(), x.max())
axs.set_ylim(y.min() - 10, y.max() + 10)
plt.show()
