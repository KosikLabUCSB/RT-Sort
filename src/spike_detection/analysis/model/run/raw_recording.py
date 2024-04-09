# Imports
from tqdm import tqdm
import numpy as np
import torch

import matplotlib.pyplot as plt

import sys
sys.path.append("/data/MEAprojects/DLSpikeSorter/src")

from model import ModelSpikeSorter
from meta import SI_MOUSE

RECORDING = np.load(SI_MOUSE[2] / "traces.npy", mmap_mode="r")
SAMP_FREQ = 30
MODEL = ModelSpikeSorter.load("/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/dl_models/230710/c/230710_172227_706810")

MODEL_OUTPUTS_PATH = MODEL.path / "log/windows_200_120/outputs.npy"  # Path to store model's outputs
WINDOWS_PER_BATCH = 100  # Each window refers to sample of (recording_num_chans, model_sample_size)

## 

num_chans, rec_duration = RECORDING.shape

start_frames_all = np.arange(0, rec_duration-MODEL.sample_size+1, MODEL.num_output_locs)

print("Allocating memory for model outputs ...")
outputs_all = np.zeros((num_chans, start_frames_all.size*MODEL.num_output_locs), dtype="float16")
np.save(MODEL_OUTPUTS_PATH, outputs_all)
outputs_all = np.load(MODEL_OUTPUTS_PATH, mmap_mode="r+")

spike_times = [[] for _ in range(num_chans)]  # Each element is list of spike times on channel
spike_amps = [[] for _ in range(num_chans)]  # Each element is list of amplitudes of spikes on each channel

print("Running model ...")
for start_idx in tqdm(range(0, start_frames_all.size, WINDOWS_PER_BATCH)):
    traces = []
    for start_frame in start_frames_all[start_idx:start_idx+WINDOWS_PER_BATCH]:
        trace = RECORDING[:, start_frame:start_frame+MODEL.sample_size]
        trace = trace - np.mean(trace, axis=1, keepdims=True)
        traces.append(trace[:, None, :])
    traces_np = np.vstack(traces).astype("float16")
    traces_torch = torch.tensor(traces_np, device=MODEL.device, dtype=MODEL.dtype)

    with torch.no_grad():
        outputs = MODEL(traces_torch).cpu()
        
    for i, pred in enumerate(MODEL.outputs_to_preds(outputs)):       
        channel = i % num_chans
        idx = start_idx + i // num_chans
        start_frame = start_frames_all[idx]

        outputs_all[channel, start_frame:start_frame+MODEL.num_output_locs] = outputs[i, :]

        if pred.size == 0:
            continue

        pred_ms = (pred + start_frame) / SAMP_FREQ
        spike_times[channel].extend(pred_ms)
        
        amps = traces_np[channel][0, pred]
        spike_amps[channel].extend(amps)
        
MODEL.log("windows_200_120/spike_times.npy", np.array(spike_times, dtype=object))
MODEL.log("windows_200_120/spike_amps.npy", np.array(spike_amps, dtype=object))