"""
Plot the average waveform shape of all units across all recordings
"""

from src.data import WaveformDataset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    root = Path(ROOT_PATH)

    wfs = []
    for rec in RECS:
        wf_dataset = WaveformDataset(root / rec / "sorted.npz", thresh_amp=THRESH_AMP, thresh_std=THRESH_STD)
        wfs += [wf_dataset[i][0] for i in range(len(wf_dataset))]
    print(f"Num wfs: {len(wfs)}")
    wfs = np.array(wfs)
    wfs_mean = np.mean(wfs, axis=0)
    plt.plot(wfs_mean)
    plt.show()


if __name__ == "__main__":
    ROOT_PATH = "/data/MEAprojects/DLSpikeSorter"
    RECS = ["2950", "2953", "2954", "2957", "5116", "5118"]
    THRESH_AMP = 3
    THRESH_STD = 0.6
    main()
