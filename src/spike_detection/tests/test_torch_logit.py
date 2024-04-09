"""
torch.logit should be the inverse of torch.sigmoid
"""

from torch import logit, sigmoid, rand
import torch
import numpy as np

x = rand(1000, dtype=torch.float32) * 200 - 100
y = logit(sigmoid(x))
close = torch.isclose(y, x)
print(f"Close/Total: {sum(close)}/{len(y)}")  # Not 100% because not enough precision, so values are rounded to 0 and 1, making inf

all = True
for thresh in np.arange(0, 1, 0.001):
    if thresh == 0:
        continue
    crossed_with_sigmoid = sum(sigmoid(x) > thresh)
    crossed_with_logit = sum(x > logit(torch.tensor(thresh)).item())
    all = all and crossed_with_sigmoid == crossed_with_logit
print(f"Num crossings with sigmoid and logit are the same: {all}")
