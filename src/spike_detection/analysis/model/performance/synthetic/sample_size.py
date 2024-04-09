"""
Test model's performance on sample sizes different from what it was trained on

Increasing sample size makes model slower by factor of how much larger sample size is
"""

from src.utils import random_seed
from src.data import RecordingCrossVal
from src.model import ModelSpikeSorter
from src.plot import set_dpi
import matplotlib.pyplot as plt
import torch


def main(sample_sizes: list,
         model_path: str, rec: str):

    # Get model
    model = ModelSpikeSorter.load(model_path).eval()
    loc_prob_thresh_og = model.get_loc_prob_thresh()

    # Get model performance based on different sample sizes
    pre_tuned_f1s = []
    post_tuned_f1s = []
    for sample_size in sample_sizes:  # Iterate through sample sizes
        model.set_loc_prob_thresh(loc_prob_thresh_og)  # Rest model's original location probability threshold

        # Get new data based on new sample size
        rec_cross_val = RecordingCrossVal(sample_size=sample_size, front_buffer=model.buffer_front_sample, end_buffer=model.buffer_end_sample,
                                          num_wfs_probs=[0.6, 0.24, 0.12, 0.04], isi_wf_min=5, isi_wf_max=None,
                                          thresh_amp=3, thresh_std=0.6,
                                          samples_per_waveform=20,
                                          device="cuda", mmap_mode="r",
                                          batch_size=10000)
        _, train, val = rec_cross_val[rec]

        # Store models inputs and outputs so perf without and with tuning are based on same data (removes difference in perf due to chance)
        inputs_list = []
        outputs_list = []
        with torch.no_grad():
            for inputs, num_wfs, wf_locs, wf_alphas in val:
                inputs_list.append((inputs, num_wfs, wf_locs, wf_alphas))
                outputs_list.append(model(inputs))

        # Performance without specific tuning
        perf = model.perf(inputs_list, outputs_list=outputs_list)
        model.perf_report(f"{sample_size} samples and {model.get_loc_prob_thresh():.1f}% thresh", perf)
        pre_tuned_f1s.append(perf[5])

        # Performance after retuning based on new sample size
        model.tune_loc_prob_thresh(train, verbose=False)
        perf = model.perf(inputs_list, outputs_list=outputs_list)
        model.perf_report(f"{sample_size} samples and {model.get_loc_prob_thresh():.1f}% thresh", perf)
        post_tuned_f1s.append(perf[5])

    # Plot data
    sample_sizes = [size / 20 for size in sample_sizes]  # Convert to frames to ms

    plt.plot(sample_sizes, pre_tuned_f1s)
    plt.scatter(sample_sizes, pre_tuned_f1s, label="pre-tuning")

    plt.plot(sample_sizes, post_tuned_f1s)
    plt.scatter(sample_sizes, post_tuned_f1s, label="post-tuning")

    plt.legend()
    plt.title("Model's performance on varying sample sizes")
    plt.xlabel("Sample size (ms)")
    plt.ylabel("F1 score (%)")
    plt.show()


if __name__ == "__main__":
    # Performance is about the same with larger sample sizes, but speed is a lot slower

    random_seed(11)
    set_dpi(400)
    main(sample_sizes=[120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
         model_path="/data/MEAprojects/DLSpikeSorter/models/v0_4_4/2954/230101_134042_729459",
         rec="2954")
