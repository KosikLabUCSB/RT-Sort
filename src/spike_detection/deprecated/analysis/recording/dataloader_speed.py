DATALOADER_KWARGS = {
    "sample_size": 200,
    "front_buffer": 20,
    "end_buffer": 40,

    "num_wfs_probs": [0.6, 0.24, 0.12, 0.04],
    "isi_wf_min": 5,
    "isi_wf_max": None,
    "thresh_amp": 3,
    "thresh_std": 0.6,
    "samples_per_waveform": 2,

    "data_root": "/data/MEAprojects/DLSpikeSorter",
    "mmap_mode": "r",
    "device": "cuda",
    "as_datasets": False,

    "batch_size": 32,
    "num_workers": 0,
    "shuffle": True
}
NUM_TESTS = 10
RECORDING_CROSS_VAL_IDX = 0

################################################################################
from src.data import RecordingCrossVal
from time import time
import torch

torch.cuda.synchronize()
recording_cross_val = RecordingCrossVal(**DATALOADER_KWARGS)
rec, train, val = recording_cross_val[RECORDING_CROSS_VAL_IDX]

print("Warming up ...")
for stuff in train:
    pass

print("Starting tests ...")
time_total = 0
for i in range(1, NUM_TESTS+1):
    time_start = time()
    for stuff in train:
        pass
    time_end = time()
    duration = time_end - time_start
    print(f"Iter {i}/{NUM_TESTS}, epoch time: {duration:.3f}s")
    time_total += duration
print(f"Avg time: {time_total / NUM_TESTS:.3f}s")
