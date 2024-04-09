from automated_detection_propagation import automated_detection_propagation
from get_inputs_from_maxwell import get_inputs_from_maxwell
from plot_propagation_signal import plot_propagation_signal


RECORDING = "data/220505/14086/000359"  # "data/220705/16460/000439"
FIRST_N_MIN = 30


from pathlib import Path
def get_recordings(path: Path, recordings: list):
    for i, p in enumerate(path.iterdir()):
        if p.is_file():
            recordings.append(path)
            break
        else:
            get_recordings(p, recordings)


if __name__ == "__main__":
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 300
    import numpy as np

    recordings = []
    get_recordings(Path("data"), recordings)
    for RECORDING in recordings:
        RECORDING = str(RECORDING)
        print(RECORDING)

        spike_times, channel_map = get_inputs_from_maxwell(RECORDING + "/data.raw.h5", first_n_min=FIRST_N_MIN)
        # np.save(RECORDING + "/spike_times.npy", spike_times)
        np.save(RECORDING + "/channel_map.npy", channel_map)

        list_of_propagation, time_all = automated_detection_propagation(spike_times,
                                                                        thres_freq=0.05,
                                                                        seconds_recording=FIRST_N_MIN * 60,
                                                                        thres_number_spikes=None,
                                                                        ratio=0.5,
                                                                        thres_cooccurrences=50,
                                                                        p=50)
        np.save(RECORDING + "/list_of_propagation.npy", np.array(list_of_propagation, dtype=object))

        plot_propagation_signal(list_of_propagation, channel_map, kilosort_npz_path=f"{RECORDING}/sorted.npz")
        break
