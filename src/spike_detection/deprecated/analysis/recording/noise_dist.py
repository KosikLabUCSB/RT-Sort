"""
Plot histogram of noise of each channel in recording
"""

from spikeinterface.extractors import MaxwellRecordingExtractor
from spikeinterface.preprocessing import scale, bandpass_filter
from spikeinterface.core import get_noise_levels
from src.data import is_dl_recording_folder
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main(rec_path_root="/data/MEAprojects/DLSpikeSorter"):
    noises = []
    for rec_path in Path(rec_path_root).iterdir():
        if not is_dl_recording_folder(rec_path): continue

        rec = MaxwellRecordingExtractor(rec_path)
        scaled = scale(rec, gain=rec.get_channel_gains(), offset=rec.get_channel_offsets(), dtype="float32")
        filt = bandpass_filter(scaled, freq_min=300, freq_max=3000, dtype="float32")
        noises = get_noise_levels(filt, return_scaled=False)

        plt.hist(noises, label=f"Median: {np.median(noises):.2f}")

        plt.title("Channel Noises")
        plt.xlabel("Noise (µV)")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
