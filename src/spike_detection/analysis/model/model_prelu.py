"""Look at alpha values of model's PReLUs"""

from src.model import ModelSpikeSorter
from torch.nn import PReLU
import torch

model = ModelSpikeSorter.load("/data/MEAprojects/DLSpikeSorter/models/v0_4/2954/rms_conv")

prelus = [module for module in model.modules() if isinstance(module, PReLU)]
for i, prelu in enumerate(prelus):
    mean = torch.mean(prelu.weight).item()
    print(f"Layer {i+1}: {mean}")
