"""
Plot distribution of calculating RMS on synthetic data samples
"""

from src.data import RecordingCrossVal
import torch
import matplotlib.pyplot as plt


def main():
    cross_val = RecordingCrossVal(sample_size=200, front_buffer=20, end_buffer=40,
                                  num_wfs_probs=[1], isi_wf_min=5, isi_wf_max=None,
                                  thresh_amp=3, thresh_std=0.6,
                                  mmap_mode="r", as_datasets=True,
                                  samples_per_waveform=100,
                                  batch_size=32, num_workers=0, shuffle=True
                                  )

    for rec, train, val in cross_val:
        cross_val.summary(rec)

        train_rms = get_rms(train)
        val_rms = get_rms(val)

        plt.hist(train_rms)
        plt.title(f"{rec} train. Mean: {train_rms.mean()}")
        plt.xlim(0, 4.5)
        plt.show()

        plt.hist(val_rms)
        plt.title(f"{rec} val. Mean: {val_rms.mean()}")
        plt.xlim(0, 4.5)
        plt.show()


def get_rms(dataset):
    data = torch.stack([dataset[i][0] for i in range(len(dataset))])
    return torch.sqrt(torch.mean(torch.square(data), dim=2)).numpy().flatten()


if __name__ == "__main__":
    main()
