# Plot a bar graph showing (num_spikes_detected/num_spikes) for different numbers of spikes in a sample (1, 2, 3, 4 spikes)

from src.model import ModelSpikeSorter
from src.data import RecordingCrossVal
from src.utils import random_seed
from src.plot import set_dpi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main(models: dict, nums_wfs: list, all_wfs_probs: list, model_path_root="/data/MEAprojects/DLSpikeSorter/models/v0_4_4",
         save_path=None):
    # model_path_root = Path(model_path_root)
    #
    # rec_cross_val = RecordingCrossVal(sample_size=None, front_buffer=None, end_buffer=None,
    #                                   num_wfs_probs=None, isi_wf_min=5, isi_wf_max=None,
    #                                   thresh_amp=3, thresh_std=0.6,
    #                                   samples_per_waveform=20,
    #                                   device="cuda", mmap_mode="r",
    #                                   batch_size=10000)
    #
    # recall_means = []
    # recall_stds = []
    # precision_means = []
    # precision_stds = []
    # for num in nums_wfs + [all_wfs_probs]:
    #     print(f"\n{num} wfs")
    #
    #     if isinstance(num, int):
    #         num_wfs_probs = [0] * (num - 1) + [1]
    #     else:
    #         num_wfs_probs = num
    #     rec_cross_val.dataset_kwargs["num_wfs_probs"] = num_wfs_probs
    #     recalls = []
    #     precisions = []
    #
    #     # Loop through each model
    #     for rec, model_name in models.items():
    #         model = ModelSpikeSorter.load(model_path_root / rec / model_name)  # Get model
    #         if rec_cross_val.dataset_kwargs["sample_size"] is None:  # Set dataset kwargs
    #             rec_cross_val.dataset_kwargs["sample_size"] = model.sample_size
    #             rec_cross_val.dataset_kwargs["front_buffer"] = model.buffer_front_sample
    #             rec_cross_val.dataset_kwargs["end_buffer"] = model.buffer_end_sample
    #
    #         rec, train, val = rec_cross_val[rec]
    #         # rec_cross_val.summary(rec)
    #
    #         model.tune_loc_prob_thresh(train, verbose=False)
    #
    #         perf = model.perf(val)
    #         model.perf_report(f"{num} wfs", perf)
    #         recalls.append(perf[3])
    #         precisions.append(perf[4])
    #
    #     recall_means.append(np.mean(recalls))
    #     recall_stds.append(np.std(recalls))
    #     precision_means.append(np.mean(precisions))
    #     precision_stds.append(np.std(precisions))

    # df = pd.DataFrame(
    #     {
    #         "Recall": recall_means,
    #         "Precision": precision_means
    #     },
    #     index=nums_wfs + ["1 to 4"]
    # )
    #
    # if save_path is not None:
    #     save_path = Path(save_path)
    #     df.to_csv(save_path / "perf.csv")
    #     np.save(save_path / "recall_stds.npy", recall_stds)
    #     np.save(save_path / "precision_stds.npy", precision_stds)

    df = pd.read_csv("/data/MEAprojects/DLSpikeSorter/models/v0_4_4/perf_multi_wf/untuned/perf.csv")
    recall_stds = np.load("/data/MEAprojects/DLSpikeSorter/models/v0_4_4/perf_multi_wf/untuned/recall_stds.npy")
    precision_stds = np.load("/data/MEAprojects/DLSpikeSorter/models/v0_4_4/perf_multi_wf/untuned/precision_stds.npy")

    df.index = ["1", "2", "3", "4", "5", "1-4"]
    df.plot.bar(rot=0, yerr=[recall_stds, precision_stds], capsize=3)
    plt.title("Performance based on number of waveforms")
    plt.xlabel("Number of waveforms in samples")
    plt.ylabel("Mean ± STD across 6 test recordings")
    plt.ylim(60, 100)
    yticks_locs, _ = plt.yticks()
    yticks_labels = [f"{p:.0f}%" for p in yticks_locs if p <= 100]
    plt.yticks(yticks_locs[:len(yticks_labels)], yticks_labels)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random_seed(501)
    set_dpi(400)
    main(models={
            "2950": "230101_133131_959516",
            "2953": "230101_133514_582221",
            "2954": "230101_134042_729459",
            "2957": "230101_134408_403069",
            "5116": "230101_134927_487762",
            "5118": "230101_135307_305876",
        },
        nums_wfs=[1, 2, 3, 4, 5],
        all_wfs_probs=[0.6, 0.24, 0.12, 0.04],
        save_path=None)
