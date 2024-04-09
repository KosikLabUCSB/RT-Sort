"""
Kilosort units have different unit IDs depending on context
    1. Index in Kilosort obj (after curation)
    2. Kilosort2 given ID (after Kilosort2 runs)

"""

from src.sorters.base import *

# for get_experts_kilosort()
from src.comparison import Comparison, DummySorter
from spikeinterface.extractors import NwbRecordingExtractor, NwbSortingExtractor


class KilosortUnit(Unit):
    def __init__(self, idx, waveforms_path,
                 id: int, spike_train, channel_idx: int, recording: Recording):
        """

        :param idx:
            In Kilosort object
        :param waveforms_path:
        :param id:
            Given by Kilosort2
        :param spike_train:
        :param channel_idx:
        :param recording:
        """

        super().__init__(id, spike_train, channel_idx, recording)

        self.idx = idx
        self.waveforms_path = Path(waveforms_path)

    # def get_waveforms(self):
    #     return channel_first(np.load(str(self.waveforms_path / f"waveforms/waveforms_{self.id}.npy"), mmap_mode="r"))

    def get_template_mean(self):
        return np.load(str(self.waveforms_path / f"templates/templates_average.npy"), mmap_mode="r")[self.id].T


class KilosortRaw(SpikeSorter):
    """Class to represent raw outputs from Kilosort2 (before spikesort_matlab4)"""
    def __init__(self, kilosort_path, recording, name="Kilosort2"):
        self.ks_path = Path(kilosort_path)

        self.spike_times = np.load(str(self.ks_path / "spike_times.npy"), mmap_mode="r").flatten()
        self.spike_clusters = np.load(str(self.ks_path / "spike_clusters.npy"), mmap_mode="r").flatten()
        self.unit_ids = np.unique(self.spike_clusters)

        self.unit_chans = None
        super().__init__(recording, name)

    def get_spike_times(self, ms=True):
        """
        Convert kilosort2's outputs to standard format

        :param ms:
            If True, return spike times in ms
            Else, return in samples

        :return [units x spike_times]: list
            Each element represents a unit and is a list of spike times

        (Not implemented yet)
            :return: [units x sequences]: list
                    Each element represents a unit and is a list of channel numbers sorted by amplitude

            ith unit in [units x sequences] corresponds to ith unit in [units x spike_times]
        """

        # spike_times = {}
        # for st, sc in zip(self.spike_times, self.spike_clusters):
        #     if ms:
        #         st /= self.recording.get_sampling_frequency()
        #     if sc in spike_times:
        #         spike_times[sc].append(st)
        #     else:
        #         spike_times[sc] = [st]
        #
        # return list(spike_times.values())
        return [u.spike_train for u in self]

    def __len__(self):
        return len(self.unit_ids)

    def __getitem__(self, idx):
        id = self.unit_ids[idx]
        spike_train = self.spike_times[self.spike_clusters == id] / self.recording.get_sampling_frequency()
        if self.unit_chans is None:
            chan = -1
        else:
            chan = self.unit_chans[id]
        return Unit(id, spike_train, chan, self.recording)


class Kilosort(KilosortRaw):
    """Class to represent data from Kilosort2 at different levels of curation"""

    def __init__(self, spikesort_matlab4_path, curation_level, recording, name="Kilosort2"):
        assert curation_level in {"initial", "first", "second"}

        self.curation_level = curation_level

        sm4_path = Path(spikesort_matlab4_path)

        self.waveforms_path = sm4_path / "waveforms"

        # self.ks_path = sm4_path / "kilosort2_results"

        # unit_id to max channel
        super().__init__(sm4_path / "curation" / curation_level, recording, name)
        self.unit_chans = np.load(str(self.waveforms_path / "channels_max/chans_max_all.npy"))

    def __getitem__(self, idx):
        unit = super().__getitem__(idx)
        return KilosortUnit(idx, self.waveforms_path,
                            -1, unit.spike_train, unit.chan, unit.recording)

    def get_templates_mean(self):
        return channel_first(np.load(str(self.waveforms_path / "templates/templates_average.npy"), mmap_mode="r"))

    def get_templates_std(self):
        return channel_first(np.load(str(self.waveforms_path / "templates/templates_std.npy"), mmap_mode="r"))

    def get_spike_times_raw(self):
        # Get spike times directly from kilosort (no recentering) as np.array
        return np.load(str(self.ks_path / "spike_times_kilosort.npy"), mmap_mode="r").flatten()


def channel_first(array):
    """

    :param array:
        Shape (N, num_samples, num_channels)
    :return:
        Shape (N, num_channels, num_samples)
    """

    return np.transpose(array, (0, 2, 1))


def load_expert_kilosort(sorting, testing_ms):
    """
    Load spike_times of expert kilosort sorting (only used in get_experts_kilosort)
    
    testing_ms:
        If not None, only use spikes between testing_ms[0] and testing_ms[1] (in ms)
    """

    spike_times = []
    for uid in sorting.get_unit_ids():
        times = sorting.get_unit_spike_train(uid) / 30


def get_experts_kilosort(freq_max=3000, overlap_time=None,
                         testing_ms=None):
    """
    Params:
    overlap_time:
        If not None, concatenate spike trains of units detected by both experts
        If a spike detected by unit1 is within overlap_time of a spike detected by unit2,
        only the spike detected by unit1 will be in the concatenated spike train
        (unit1 in this case is the unit that first detects the spike)
    
    testing_ms:
        If not None, only use spikes between testing_ms[0] and testing_ms[1] (in ms)
    """
    
    sorting1 = NwbSortingExtractor("/data/MEAprojects/dandi/000034/sub-mouse412804/sub-mouse412804_ses-20200824T155542.nwb", sampling_frequency=30000)
    sorting2 = NwbSortingExtractor("/data/MEAprojects/dandi/000034/sub-mouse412804/sub-mouse412804_ses-20200824T155543.nwb", sampling_frequency=30000)
    
    # start_ms, end_ms = testing_ms
    # spikes1 = []
    # for times in 
    
    spikes1 = [sorting1.get_unit_spike_train(uid) / 30 for uid in sorting1.get_unit_ids()]
    spikes2 = [sorting2.get_unit_spike_train(uid) / 30 for uid in sorting2.get_unit_ids()]

    sorter1 = DummySorter(spikes1)
    sorter2 = DummySorter(spikes2)
    comp = Comparison(sorter1, sorter2)
    
    # Use only the spike times from expert 1
    if overlap_time is None:
        spike_times = np.array([spikes1[idx] for idx in range(len(comp.match12)) if comp.match12[idx] != -1])
    else:  # Concatenate the spike times from experts
        spike_times = []
        for idx1, idx2 in enumerate(comp.match12):
            if idx2 == -1:
                continue
            
            # Copied and pasted from si_rec6.py:
            spike_train_i = spikes1[idx1]
            spike_train_j = spikes2[idx2]
            
            spike_train = [spike_train_i[0]]
            i, j = 1, 0
            while i < len(spike_train_i) and j < len(spike_train_j):
                spike_i, spike_j = spike_train_i[i], spike_train_j[j]
                if spike_i < spike_j:  # i is next to be added
                    if np.abs(spike_train[-1] - spike_i) >= overlap_time:  # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
                        spike_train.append(spike_i)
                    i += 1
                else:  # j is next to be added
                    if np.abs(spike_train[-1] - spike_j) >= overlap_time: # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
                        spike_train.append(spike_j)
                    j += 1

            # Append remaning elements (only one cluster's spike train can be appended due to while loop)
            if i < len(spike_train_i):
                spike_train.extend(spike_train_i[i:])
            else:
                spike_train.extend(spike_train_j[j:])
            
            spike_times.append(spike_train)

    if testing_ms is not None:
        start_ms, end_ms = testing_ms
        testing_spike_times = []
        for times in spike_times:
            times = np.array(times)
            times_ind = start_ms <= times
            times_ind *= times <= end_ms
            testing_spike_times.append(times[times_ind])
        spike_times = testing_spike_times

    sorter = SpikeSorter(
        Recording("/data/MEAprojects/dandi/000034/sub-mouse412804/sub-mouse412804_ecephys.nwb", freq_max=freq_max),
        name="Experts Kilosort",
        spike_times=spike_times
    )
    
    return sorter

