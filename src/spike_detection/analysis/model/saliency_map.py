"""
Create saliency map of training samples to visualize what model looks for in input
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from src.model import ModelSpikeSorter
from src.data import MultiRecordingDataset
from src import plot, utils


MODEL_PATH = r"/data/MEAprojects/DLSpikeSorter/models/v0_3/2954"
REC = "2954"
Y_BUFFER = 5


def main():
    plot.set_dpi(400)
    utils.random_seed(231)

    # Get model
    model = ModelSpikeSorter.load(MODEL_PATH)
    model.eval()
    dataset = MultiRecordingDataset.load_single(path_folder="/data/MEAprojects/DLSpikeSorter/" + REC,
                                                samples_per_waveform=2, front_buffer=model.buffer_front_sample, end_buffer=model.buffer_end_sample,
                                                num_wfs_probs=[1], isi_wf_min=5, isi_wf_max=None,
                                                sample_size=model.sample_size,
                                                device="cuda", dtype=torch.float32,
                                                thresh_amp=12, thresh_std=0.6)

    for i in range(1, len(dataset), 2):
        fig, axs = plt.subplots(1, figsize=(6, 6), tight_layout=True)

        trace, num_wf, wf_locs, wf_alphas = dataset[i]
        trace.requires_grad = True

        output = model(trace[None, :])
        pred_locs = model.outputs_to_preds(output.detach())[0]

        # Derivative of input wrt wrt_loc will be found
        if len(pred_locs) > 0:
            wrt_loc = pred_locs[0]
            # wf_prob = torch.sigmoid(output[0, model.logit_to_loc(wrt_loc)]).detach().cpu().item()
        else:
            wrt_loc = model.sample_size // 2


        output[0, model.loc_to_logit(wrt_loc)].backward()

        grad = trace.grad[0, :].cpu().numpy()
        trace = trace[0, :].detach().cpu().numpy()
        output = output[0, :].detach().cpu().numpy()

        trace_x = np.arange(trace.size)  # np.linspace(0, trace.size, 10000)
        trace_interp = trace  # np.interp(trace_x, np.arange(trace.size), trace)
        trace_points = np.array([trace_x, trace_interp]).T.reshape(-1, 1, 2)
        trace_segments = np.concatenate([trace_points[:-1], trace_points[1:]], axis=1)

        # grad_interp = np.interp(
        #     np.linspace(0, grad.size, 10000),
        #     np.arange(grad.size),
        #     grad
        # )
        grad_interp = grad

        trace_lc = LineCollection(trace_segments, cmap="viridis", linewidths=2)
        trace_lc.set_array(grad_interp)
        line = axs.add_collection(trace_lc)
        fig.colorbar(line, ax=axs)

        # if len(pred_)
        # plot.display_prob_spike(output[model.idx_spike], axs)

        axs.set_title(f"Sample")
        axs.axvline(wrt_loc, linestyle="dashed", color="red", alpha=0.5, label="Model")
        for loc in wf_locs.cpu().numpy():
            axs.axvline(loc, linestyle="dashed", color="blue", alpha=0.5, label="Label")
        axs.legend(loc="upper right")

        axs.set_xlim(0, trace.size-1)
        axs.set_xlabel(plot.TRACE_X_LABEL)
        axs.set_ylim(trace.min() - Y_BUFFER, trace.max() + Y_BUFFER)

        plt.show()


if __name__ == "__main__":
    main()
