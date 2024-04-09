"""
Test src.data.BandpassFilter
"""

ROOT = "/data/MEAprojects/DLSpikeSorter"
REC = "2954"
FREQ_BAND = (300, 6000)
FILTER_TYPE = "bandpass"

from src.data import BandpassFilter, MultiRecordingDataset
import numpy as np
from pathlib import Path
from spikeinterface import preprocessing
from spikeinterface.extractors import MaxwellRecordingExtractor
import matplotlib.pyplot as plt
import torch


def compare_with_si(freq_min, freq_max,
                    start_frame, duration, channel_idx):
    rec_path = Path(ROOT) / REC
    end_frame = start_frame + duration

    rec_si = MaxwellRecordingExtractor(rec_path / "data.raw.h5")
    filt_si = preprocessing.bandpass_filter(rec_si, freq_min=freq_min, freq_max=freq_max, dtype="float32")  # type: MaxwellRecordingExtractor

    rec_dl = np.load(rec_path / "data.raw.npy", mmap_mode="r")
    bandpass_filter = BandpassFilter((freq_min, freq_max))

    channel_id = filt_si.get_channel_ids()[channel_idx]
    trace_raw_si = rec_si.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=[channel_id])
    trace_filt_si = filt_si.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=[channel_id])

    trace_raw_dl = rec_dl[channel_idx, start_frame:end_frame]
    trace_filt_dl = bandpass_filter(trace_raw_dl)

    fig, (a0, a1) = plt.subplots(2, 1, tight_layout=True)

    a0.set_title("Raw")
    a0.plot(trace_raw_si, label="SI")
    a0.plot(trace_raw_dl, label="DL")

    a1.set_title("Filtered")
    a1.plot(trace_filt_si, label="SI")
    a1.plot(trace_filt_dl, label="DL")

    plt.legend()
    plt.show()


def test_synthetic():
    # Test filter on synthetic data

    dataset = MultiRecordingDataset.load_single(path_folder="/data/MEAprojects/DLSpikeSorter/" + REC,
                                                samples_per_waveform=2, front_buffer=20, end_buffer=40,
                                                num_wfs_probs=[1], isi_wf_min=5, isi_wf_max=None,
                                                sample_size=200,
                                                device="cpu", dtype=torch.float32,
                                                thresh_amp=3, thresh_std=0.6)
    bandpass_filter = BandpassFilter(FREQ_BAND, btype=FILTER_TYPE)

    for i in range(len(dataset)):
        trace, num_wfs, wf_locs, wf_alphas = dataset[i]
        trace = trace.numpy().flatten()

        fig, (a0, a1) = plt.subplots(2)
        a0.plot(trace)
        a1.plot(bandpass_filter(trace))
        plt.show()


def main():
    test_synthetic()


if __name__ == "__main__":
    main()
