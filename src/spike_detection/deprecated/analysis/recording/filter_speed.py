"""
Test speed of bandpass filter
"""

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1020, 1024, 2048]
N_RUNS = 100

FREQ_BAND = (300, 6000)
FILTER_TYPE = "bandpass"

from src.data import BandpassFilter, MultiRecordingDataset
from time import time
import torch
from src.utils import random_seed


def main():
    dataset = MultiRecordingDataset.load_single(path_folder="/data/MEAprojects/DLSpikeSorter/2954",
                                                samples_per_waveform=1000, front_buffer=20, end_buffer=40,
                                                num_wfs_probs=[1], isi_wf_min=5, isi_wf_max=None,
                                                sample_size=200,
                                                device="cpu", dtype=torch.float32,
                                                thresh_amp=3, thresh_std=0.6)
    bandpass = BandpassFilter(FREQ_BAND, btype=FILTER_TYPE)

    random_seed(231)
    inputs_all = torch.stack([dataset[i][0] for i in range(max(BATCH_SIZES))])
    for size in BATCH_SIZES:
        inputs = inputs_all[:size]
        time_total = 0
        for _ in range(N_RUNS):
            time_start = time()
            for trace in inputs:
                bandpass(trace)
            time_end = time()
            time_total += time_end - time_start
        print(f"{size} samples: {time_total / N_RUNS:.4f}")


if __name__ == "__main__":
    main()
