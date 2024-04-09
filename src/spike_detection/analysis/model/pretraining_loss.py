"""
Get expected value of loss of model before training
"""

NUM_OUTPUT_LOCS = 140
NUM_WF_PROBS = [0.6, 0.24, 0.12, 0.04]

#######################################
import torch

bce = torch.nn.BCEWithLogitsLoss()

y_hat = [0] * 10
y = [1, 0] * 5

y_hat = torch.tensor(y_hat, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

loss = bce(y_hat, y)
print(loss)
