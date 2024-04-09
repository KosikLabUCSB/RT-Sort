import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 500
from pandas import DataFrame
import numpy as np
from pathlib import Path
import json

DATA_PATH = "data"
ISI_PERCENTS = [0.15, 1, 2, 100]
ISI_THRESHOLD_SAMPLES = 1.5 * 20


class Recording:
    def __init__(self, path):
        self._rec = path
        self.name = f"{path.parent.parent.name}/{path.parent.name}"

    def __repr__(self):
        return str(self._rec)

    def get_num_propagation(self):
        return len(np.load(str(self._rec / "list_of_propagation.npy"), allow_pickle=True).tolist())

    def get_spike_times(self):
        return np.load(str(self._rec / "spike_times.npy"))

    def get_spike_clusters(self):
        return np.load(str(self._rec / "spike_clusters.npy"))

    def get_curation_history(self):
        with open(self._rec / "curation_history.json", "r") as j:
            return json.load(j)

    @staticmethod
    def get_recordings(path: Path, recordings: list):
        for i, p in enumerate(path.iterdir()):
            if p.is_file():
                recordings.append(Recording(path))
                break
            else:
                Recording.get_recordings(p, recordings)


def main():
    recordings = []
    Recording.get_recordings(Path(DATA_PATH), recordings)
    bar_frame_init = {"Destinee": []}
    for p in ISI_PERCENTS:
        bar_frame_init[str(p) + "%"] = []
    bar_frame = DataFrame(bar_frame_init)

    for rec in recordings:  # type: Recording
        rec_bar = [rec.get_num_propagation()]

        history = rec.get_curation_history()
        none = set(history["initial"])
        for curation in ["fr", "snr", "spikes_min_first"]:
            none.intersection_update(history["curated"][curation])

        spike_times = rec.get_spike_times()
        spike_clusters = rec.get_spike_clusters()

        for isi in ISI_PERCENTS:
            n_curated = 0
            for unit in none:
                spike_train = spike_times[np.flatnonzero(spike_clusters == unit)]
                num_spikes = spike_train.size

                isis = np.diff(spike_train)
                violation_num = np.sum(isis < ISI_THRESHOLD_SAMPLES)
                violation_percent = (violation_num / num_spikes) * 100
                if violation_percent <= isi:
                    n_curated += 1
            rec_bar.append(n_curated)

        print(rec.name)
        print(rec_bar)
        exit()
        bar_frame.loc[rec.name] = rec_bar

    bar_frame.plot.bar(rot=15)
    plt.xlabel("Recording")
    plt.ylabel("Number of units")
    plt.yticks(np.arange(0, 20, 2))
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
