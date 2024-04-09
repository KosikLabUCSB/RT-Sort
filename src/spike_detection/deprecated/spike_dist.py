"""
Using kilosort's spike times, plot a distribution of the number of spikes in each sliding window

1. Create spike_array of shape (n_channels, n_samples) where each element is 0 except when there is a spike (1)
    a. Loop through each unit, add 1s where spike occurs and amplitude of template on channel is greater than a threshold
2. Loop through each window of array and count spikes
"""


REC = 2953
SPIKE_TIMES_PATH = f"/data/MEAprojects/DLSpikeSorter/{REC}/spikesort_matlab4/first_curation/spike_times.npy"
SPIKE_CLUSTERS_PATH = f"/data/MEAprojects/DLSpikeSorter/{REC}/spikesort_matlab4/first_curation/spike_clusters.npy"
TEMPLATES_MEAN_PATH = f"/data/MEAprojects/DLSpikeSorter/{REC}/spikesort_matlab4/templates/templates_average.npy"
SAVE_PATH = f"/data/MEAprojects/DLSpikeSorter/{REC}/spikesort_matlab4/spike_counts.npy"

AMP_THRESH = 3  # Minimum amplitude threshold for classifying a unit as spiking on a channel
REC_DURATION = 180 * 20000  # In samples

WINDOW_SIZE = 200
STRIDE = 1


import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from tqdm import tqdm


def get_spike_array(spike_times, spike_clusters, templates_mean, amp_thresh, rec_duration):
    # Get np.array of shape (n_samples, n_channels)

    n_chans = templates_mean.shape[2]
    spike_array = np.zeros((rec_duration, n_chans), dtype=bool)

    # Iterate through units
    unit_ids = np.unique(spike_clusters)
    print("Creating spike array")
    for uid in tqdm(unit_ids):
        # Get which channels unit spikes on
        templates = templates_mean[uid, :, :]
        amplitudes = np.max(np.abs(templates), axis=0)
        chans = np.flatnonzero(amplitudes >= amp_thresh)

        # Get spike train
        uid_st = spike_times[spike_clusters == uid]

        # Add to array
        for c in chans:
            spike_array[uid_st, c] = 1

    return spike_array


def count_spikes(spike_array, window_size, stride, save_path=None):
    n_samples = spike_array.shape[0]
    spike_counts = np.zeros((window_size+1,), dtype=int)  # ith element = number of windows with i spikes
    print("Counting spikes")
    for i in tqdm(range(0, n_samples-window_size, stride)):
        window = spike_array[i:i+window_size, :]
        counts = np.sum(window, axis=0)

        if np.any(counts > 4):
            print(i, np.flatnonzero(counts > 4))

        spike_counts[counts] += 1

    if save_path is not None:
        np.save(SAVE_PATH, spike_counts)
    return spike_counts


def main():
    # Get data
    spike_times = np.load(SPIKE_TIMES_PATH)
    spike_clusters = np.load(SPIKE_CLUSTERS_PATH)
    templates_mean = np.load(TEMPLATES_MEAN_PATH, mmap_mode="r")

    # Get spike counts
    # spike_array = get_spike_array(spike_times, spike_clusters, templates_mean, AMP_THRESH, REC_DURATION)
    # spike_counts = count_spikes(spike_array, WINDOW_SIZE, STRIDE, SAVE_PATH)
    spike_counts = np.load(SAVE_PATH)
    max_spikes = np.flatnonzero(spike_counts)[-1]
    spike_counts = spike_counts[:max_spikes+1] / np.sum(spike_counts) * 100

    # Plot spike counts
    df = DataFrame(
        {
            "Percent": spike_counts
        },
        index=range(len(spike_counts))
    )
    df.plot.bar(rot=0)
    plt.title(f"Num {WINDOW_SIZE}-sample windows, stride {STRIDE}")
    plt.xlabel("Number of spikes in window")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    for n, c in enumerate(spike_counts):
        print(f"{n} spikes: {c:.1f}%")


if __name__ == "__main__":
    main()


# spike_times = np.load(SPIKE_TIMES_PATH)
# spike_counts = {}
#
# for i in range(REC_DURATION-WINDOW_SIZE):
#     i0 = np.searchsorted(spike_times, i)
#     i1 = np.searchsorted(spike_times, i+WINDOW_SIZE)
#     num_spikes = spike_times[i0:i1].size
#     if num_spikes not in spike_counts:
#         spike_counts[num_spikes] = 1
#     else:
#         spike_counts[num_spikes] += 1
#
# spike_counts_np = np.array([spike_counts[c] for c in range(max(spike_counts.keys()))])
# plt.bar(range(spike_counts_np.size), spike_counts_np / sum(spike_counts_np))
# plt.show()

