from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import pad as torch_pad
import torch_tensorrt
from scipy.signal import find_peaks

from src.model import ModelSpikeSorter
from src import utils
# test = torch.rand((1020, 120), dtype=torch.float16, device="cuda")
#
# start = perf_counter()
# test = test.cpu()
# for x in test:
#     y = find_peaks(x)
# end = perf_counter()
# print((end - start) * 1000)

RECORDING = np.load(utils.PATH_REC_DL_NP.format("2953"), mmap_mode="r")
MODEL = ModelSpikeSorter.load("/data/MEAprojects/DLSpikeSorter/models/v0_4_4/2953/230101_133514_582221").eval().cuda().to(torch.float16)

##

prop_array = [0] * 1020
prop_array2 = [[0]]

loc_prob_thresh_logit = MODEL.loc_prob_thresh_logit
logit_to_loc_add = MODEL.loc_first_frame

traces = RECORDING[:, 2000:2200]

traces = traces - np.mean(traces, axis=1, keepdims=True)
traces = torch.tensor(traces[:, None, :], device="cuda", dtype=torch.float16)

def warmup(model, inputs, n_runs, verbose=False):
    # GPU needs to warmup

    if verbose:
        print("Warming up ...")
    with torch.no_grad():
        for _ in range(n_runs):
            model(inputs.to("cuda"))
            torch.cuda.synchronize()
warmup(MODEL, traces, 100)

MODEL = MODEL.model.conv  # Get rid of flatten layer

heavy = torch.tensor([0.5], dtype=torch.float16, device="cuda")
processing_time_start = perf_counter()
with torch.no_grad():
    outputs = MODEL(traces)
    outputs = torch.gt(outputs, 0.3)
    outputs = torch.sum(outputs, )

processing_time_end = perf_counter()

# processing_time_start = perf_counter()
#
# for _ in range(10):
#     # outputs = outputs.cpu()
#     # processing_time_end = perf_counter()
#
#     outputs = torch_pad(outputs.cpu(), (1, 1), value=-np.inf)
#     for channel in outputs.cpu():
#         peaks = find_peaks(channel, height=loc_prob_thresh_logit)[0]
#         for p in peaks:
#             prop_array[prop_array2[0][0]] = p
#
# processing_time_end = perf_counter()

print((processing_time_end - processing_time_start) * 1000)
