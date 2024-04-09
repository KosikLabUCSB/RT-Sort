from src.data import RecordingCrossVal
from src.model import ModelSpikeSorter
from src.utils import random_seed
from src.plot import set_dpi
import torch


def main():
    random_seed(231)  # 0.6, 0.24, 0.12, 0.04

    rec_cross_val = RecordingCrossVal(sample_size=200, front_buffer=40, end_buffer=40,
                                      num_wfs_probs=[0.6, 0.24, 0.12, 0.04], isi_wf_min=3, isi_wf_max=None,
                                      thresh_amp=3, thresh_std=0.6,
                                      samples_per_waveform=1, mmap_mode="r",
                                      device="cuda", dtype=torch.float16, as_datasets=False,
                                      batch_size=1000)
    # model = ModelSpikeSorter(1, 1000, 440, 440, 50, 0, 0, "cpu", None)
    model = ModelSpikeSorter.load("/data/MEAprojects/DLSpikeSorter/models/v0_4_4/2954/230101_134042_729459").to(torch.float16)

    set_dpi(400)

    rec, train, val = rec_cross_val["2954"]
    perf = model.perf(val, plot_preds=("no"))  # plot_preds=("correct")
    model.perf_report("2954", perf)

    # trace, num_wfs, wf_locs, alphas = val[0]
    # output = model(trace[:, None, :])
    # pred = model.outputs_to_preds(output)[0]
    # model.plot_pred(trace, output[0], pred, num_wfs, wf_locs, alphas)

    # perf = model.perf(val, plot_preds=("failed"), max_plots=None)


if __name__ == "__main__":
    main()
