# si = spikeinterface

from spikeinterface.extractors import NwbRecordingExtractor, MaxwellRecordingExtractor
from spikeinterface.preprocessing import highpass_filter, bandpass_filter, scale
from spikeinterface.core import get_noise_levels
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool


def get_channel_map(nwb_path: str):
    """

    :param nwb_path:
        Path to .nwb file

    :return channel_map: dict
        {channel_id: (x, y)}
    """
    rec = NwbRecordingExtractor(nwb_path)
    return {chan_id: (x, y) for chan_id, (x, y) in zip(rec.get_channel_ids(), rec.get_channel_locations())}


class GetChunkCrossingsWrapper:
    """
    Wrapper class for func in multiprocessing for getting threshold crossings of a channel

    The bottleneck for getting threshold crossings is get_traces, so this is what is being parallel processed
    """

    def __init__(self, rec_path, freq_min, freq_max, spike_amp_thresh, chunk_size):
        self.rec_path = rec_path
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.thresh = spike_amp_thresh
        self.chunk_size = chunk_size

    def __call__(self, start_frame):
        rec_path = self.rec_path
        if rec_path.suffix == ".h5":
            rec = MaxwellRecordingExtractor(rec_path)
        else:
            rec = NwbRecordingExtractor(rec_path)
        rec = scale(rec, rec.get_channel_gains(), rec.get_channel_offsets())
        if self.freq_max >= rec.get_sampling_frequency() / 2:
            # The frequency of a constituent signal must be less than the sampling_frequency / 2
            rec = highpass_filter(rec, freq_min=self.freq_min, dtype="float32")  # type: NwbRecordingExtractor
        else:
            rec = bandpass_filter(rec, freq_min=self.freq_min, freq_max=self.freq_max, dtype="float32")  # type: NwbRecordingExtractor

        fs = rec.get_sampling_frequency() / 1000
        traces_all = rec.get_traces(start_frame=start_frame, end_frame=start_frame + self.chunk_size, return_scaled=False)

        crossings = []
        for trace in traces_all.T:
            noise = np.sqrt(np.mean(trace ** 2))
            # st = find_peaks(np.abs(trace), height=self.thresh * noise)[0]
            st = find_peaks(-trace, height=self.thresh * noise)[0]  # Only negative threshold crossings

            if st.size > 0:
                st = (st + start_frame) / fs

                crossings.append(st)

            # import matplotlib.pyplot as plt
            #
            # if st.size > 0:
            #     plt.plot(trace)
            #     plt.axhline(-self.thresh * noise)
            #     plt.axhline(self.thresh * noise)
            #     for x in find_peaks(np.abs(trace), height=self.thresh * noise)[0]:
            #         plt.axvline(x, color="black", linestyle="dashed")
            #     plt.show()

        return crossings


def save_inputs(save_path: Path, rec_path: Path, freq_min=300, freq_max=3000, spike_amp_thresh=5, chunk_size=1000):
    """
    Save inputs to automated_detection_propagation from NWB file containing raw LFP with the following method:
        1. Filter the LFP between (freq_min, freq_max) in Hz
        2. A timepoint is considered a spike time if the amplitude of the signal at that point is spike_amp_thresh times the RMS of the surrounding signal

    :param save_path:
        Path to save .npy input data
    :param rec_path:
        Path to .h5 file from Maxwell Biosystems or .nwb file containing LFP data that is used for algorithm inputs
    :param freq_min:
        In Hz, filter LFP between (freq_min, freq_max)
    :param freq_max:
        In Hz, filter LFP between (freq_min, freq_max)
    :param spike_amp_thresh:
        A timepoint is considered a spike time if the amplitude of the signal at that point is spike_amp_thresh times the RMS of the surrounding signal
    :param chunk_size:

    :return spike_times: list
        Input to automated_detection_propagation (in ms).
    """

    rec_path = Path(rec_path)
    if rec_path.suffix == ".h5":
        rec = MaxwellRecordingExtractor(rec_path)
    else:
        rec = NwbRecordingExtractor(rec_path)

    spike_times = [[] for _ in range(rec.get_num_channels())]
    with Pool(processes=16) as pool:
        tasks = range(0, rec.get_total_samples(), chunk_size)
        for crossings in tqdm(pool.imap(GetChunkCrossingsWrapper(rec_path, freq_min, freq_max, spike_amp_thresh, chunk_size), tasks, len(tasks) // 40), total=len(tasks)):
            for i in range(len(crossings)):
                spike_times[i].extend(crossings[i])

    np.save(save_path, spike_times)


def main():
    save_inputs(THRESH_CROSSINGS_PATH, REC_PATH, spike_amp_thresh=SPIKE_AMP_THRESH)

    # Speed test between reading from /dev/nvme0n1p3 and /dev/sda1 --> Same speed
    # nvme = NwbRecordingExtractor("/home/mea/SpikeSorting/prop_signal/sub-c1_ecephys.nwb")
    # sda = NwbRecordingExtractor(NWB_PATH)
    #
    # start_frame = int(0)
    # end_frame = start_frame + int(sda.get_total_samples())
    #
    # from time import perf_counter
    #
    # start = perf_counter()
    # sda.get_traces(start_frame=start_frame, end_frame=end_frame)
    # end = perf_counter()
    # print(end - start)
    #
    # start = perf_counter()
    # nvme.get_traces(start_frame=start_frame, end_frame=end_frame)
    # end = perf_counter()
    # print(end - start)
    #
    # start = perf_counter()
    # sda.get_traces(start_frame=start_frame, end_frame=end_frame)
    # end = perf_counter()
    # print(end - start)
    #
    # start = perf_counter()
    # nvme.get_traces(start_frame=start_frame, end_frame=end_frame)
    # end = perf_counter()
    # print(end - start)
    #
    # start = perf_counter()
    # sda.get_traces(start_frame=start_frame, end_frame=end_frame)
    # end = perf_counter()
    # print(end - start)


SPIKE_AMP_THRESH = 5
FREQ_MIN = 300
FREQ_MAX = 3000
# ROOT = Path("/data/MEAprojects/dandi/000034/sub-c1")
# REC_PATH = ROOT / "sub-c1_ecephys.nwb"
# THRESH_CROSSINGS_PATH = ROOT / f"thresh_crossings_{SPIKE_AMP_THRESH}.npy"

# THRESH_CROSSINGS_PATH = f"/data/MEAprojects/organoid/intrinsic/220318/14086/Network/000359/thresh_crossings_{SPIKE_AMP_THRESH}.npy"
# REC_PATH = "/data/MEAprojects/organoid/intrinsic/220318/14086/Network/000359/data.raw.h5"

# THRESH_CROSSINGS_PATH = f"/data/MEAprojects/dandi/000034/sub-mouse412804/thresh_crossings_{SPIKE_AMP_THRESH}_neg_only.npy"
# REC_PATH = f"/data/MEAprojects/dandi/000034/sub-mouse412804/sub-mouse412804_ecephys.nwb"
THRESH_CROSSINGS_PATH = "/data/MEAprojects/PropSignal/data/220705_16460_000443"
REC_PATH = "/data/MEAprojects/organoid/intrinsic/220705/16460/Network/000443/data.raw.h5"

import os
os.environ['HDF5_PLUGIN_PATH'] = '/home/mea/SpikeSorting/spikeinterface/'

if __name__ == "__main__":
    main()
