import numpy as np
from pathlib import Path


class Kilosort:
    """Class to represent data from Kilosort2 at different levels of curation"""

    KILOSORT_PATH = "/data/MEAprojects/DLSpikeSorter/{recording_name}/spikesort_matlab4/kilosort2_results"
    CURATION_PATH = "/data/MEAprojects/DLSpikeSorter/{recording_name}/spikesort_matlab4/curation/{curation_level}"
    WAVEFORMS_PATH = "/data/MEAprojects/DLSpikeSorter/{recording_name}/spikesort_matlab4/waveforms"

    def __init__(self, recording_name: str, curation_level: str):
        self.recording_name = recording_name
        self.curation_level = curation_level

        curation_path = Path(Kilosort.CURATION_PATH.format(recording_name=recording_name, curation_level=curation_level))
        self.spike_times = np.load(str(curation_path / "spike_times.npy"), mmap_mode="r").flatten()
        self.spike_clusters = np.load(str(curation_path / "spike_clusters.npy"), mmap_mode="r").flatten()
        self.unit_ids = np.unique(self.spike_clusters)

        self.waveforms_path = Path(Kilosort.WAVEFORMS_PATH.format(recording_name=recording_name))
        self.templates_mean = np.load(str(self.waveforms_path / "templates/templates_average.npy"), mmap_mode="r")

        self.ks_path = Path(Kilosort.KILOSORT_PATH.format(recording_name=recording_name))

    def get_templates_std(self):
        return np.load(str(self.waveforms_path / "templates/templates_std.npy"), mmap_mode="r")

    def curate_spike_times(self, max_chan_only=False, thresh_amp=None, thresh_std=None, thresh_rms=None):
        """
        Parameters are all np.ndarray outputs of spikesort_matlab.py
        Allows spike times from different stages (kilosort, first curation, second curation) to be retrieved

        :param max_chan_only:
            If True, only use maximum channel of kilosort spikes regardless of other thresholds
        :param thresh_amp
            If None, don't thresh
            Same as always for this whole project
        :param thresh_std
            If None, don't thresh
            Same as always
        :param thresh_rms:
            If None, don't thresh
            Only count units whose peak or trough crosses thresh_rms RMS of the template

        :return: list
            Kilosort's spike times in standard format (ith element contains a list of spikes detected on ith channel)
        """
        # Dict of unit ID to spike train
        uid_to_st = {}
        for st, uid in zip(self.spike_times, self.spike_clusters):
            if uid not in uid_to_st:
                uid_to_st[uid] = [st]
            else:
                uid_to_st[uid].append(st)

        spike_times = [[] for _ in range(self.templates_mean.shape[2])]
        for uid in self.unit_ids:
            # Curate channels
            template_mean = self.templates_mean[uid, :, :]  # (n_samples, n_channels)
            chans = np.ones((template_mean.shape[1],), dtype=bool)

            amps = np.max(np.abs(template_mean), axis=0)
            if max_chan_only:
                chan_max = np.argmax(amps)
                spike_times[chan_max].extend(uid_to_st[uid])
            else:
                if thresh_amp is not None:
                    chans &= amps >= thresh_amp
                if thresh_rms is not None:
                    rmses = np.sqrt(np.mean(np.square(template_mean), axis=0))
                    chans &= amps >= (thresh_rms * rmses)
                if thresh_std is not None:
                    amps_ind = np.argmax(template_mean, axis=0)
                    std_norms = self.get_templates_std()[uid, amps_ind, range(amps_ind.size)] / amps
                    chans &= std_norms <= thresh_std

                if len(chans) > 0:
                    unit_spike_train = uid_to_st[uid]
                    for chan, is_curated in enumerate(chans):
                        if not is_curated: continue
                        spike_times[chan].extend(unit_spike_train)

        return [sorted(st) for st in spike_times]

    def get_spike_times_raw(self):
        # Get spike times directly from kilosort (no recentering) as np.array
        return np.load(str(self.ks_path / "spike_times_kilosort.npy"), mmap_mode="r").flatten()
