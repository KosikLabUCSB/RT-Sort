"""
Test the different in speed when calculating the loss

1) Indexing the samples with a spike
2) Multiplying by the labels

Conclusion:
Multiplying is MUCH faster (especially when using GPU)
"""

import torch
from time import perf_counter

data = torch.arange(100000).cuda()
labels = torch.randint(2, (len(data),)).cuda()

start = perf_counter()
out = data[labels] # data[torch.nonzero(labels)]
print(perf_counter() - start)

start = perf_counter()
out = data * labels
print(perf_counter() - start)
