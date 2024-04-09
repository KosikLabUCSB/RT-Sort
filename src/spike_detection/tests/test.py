import numpy as np
import torch
import torch_tensorrt
import time

torch.cuda.synchronize()

x = torch.rand(1, 1020*21, dtype=torch.float16, device="cuda")
# w = torch.nn.Linear(1020*21, 100, dtype=torch.float16, device="cuda")
model = torch.nn.Sequential(torch.nn.Linear(1020*21, 100))
model = model.to(dtype=torch.float16).cuda()

model = torch.jit.trace(model, x)
model = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((1, 1020*21), dtype=torch.float16)], enabled_precisions={torch.float16})

torch.cuda.synchronize()
start = time.time()
cc = model(x)
end = time.time()
print(end - start)
