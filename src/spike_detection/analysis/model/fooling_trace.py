"""
Modify a trace until the model no longer classifies it as a spike
"""

from src.model import ModelSpikeSorter
from src.data import MultiRecordingDataset
from src.plot import set_ticks
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import torch
import numpy as np
from pathlib import Path

MODEL_PATH = r"/data/MEAprojects/DLSpikeSorter/models/v0_3/"
REC = "2954"  # Which rec to get traces from
NUM_ITER = 10000  # Number of iterations for gradient descent
LR = 0.1  # Learning rate for gradient ascent
PROB_THRESHOLD = 0.2  # Threshold of probability (in decimal form, NOT percent) to stop updating trace (threshold to consider model being fooled)


def main():
    model = ModelSpikeSorter.load(Path(MODEL_PATH) / REC)
    dataset = MultiRecordingDataset.load_single(path_folder="/data/MEAprojects/DLSpikeSorter/" + REC,
                                                samples_per_waveform=2, front_buffer=model.buffer_front_sample, end_buffer=model.buffer_end_sample,
                                                num_wfs_probs=[1], isi_wf_min=5, isi_wf_max=None,
                                                sample_size=model.sample_size,
                                                device="cuda", dtype=torch.float32,
                                                thresh_amp=12, thresh_std=0.6)

    for idx in range(1, len(dataset), 2):
        trace_og, _, wf_loc, _ = dataset[idx]
        trace = trace_og.clone()[None, :].requires_grad_()
        trace_og = trace_og.cpu().numpy()[0, :]
        wf_loc = wf_loc[0]
        wf_logit = model.loc_to_logit(wf_loc)

        prob_first = None
        loc_first = None
        for i in range(NUM_ITER):
            trace.grad = None

            output = model(trace)

            prob = torch.sigmoid(output[0, wf_logit]).item()
            if prob_first is None:
                prob_first = prob
                loc_first = model.outputs_to_preds(output.detach())[0][0]
            print(f"Iter {i}: {prob :.2f}")
            if prob < PROB_THRESHOLD:
                break

            output[0, wf_logit].backward()
            with torch.no_grad():
                trace -= LR * trace.grad

        with torch.no_grad():
            output = model(trace)

            pred = model.outputs_to_preds(output)[0]
            output = output[0, :]
            prob = torch.sigmoid(output[wf_logit])
            trace = trace[0, 0, :].detach().cpu().numpy()

        fig, (a0, a1, a2) = plt.subplots(3, tight_layout=True, figsize=(6, 8))  # type: Axes

        set_ticks((a0, a1, a2), trace_og)

        a0.set_title(f"Initial probability of spike: {prob_first * 100:.2f}%")
        a0.plot(trace_og)
        a0.axvline(loc_first, linestyle="dashed", color="red", alpha=0.3, label="Model")
        a0.legend()

        a1.set_title(f"Probability of spike: {prob * 100:.2f}%")
        a1.plot(trace)
        # a1.axvline(pred[0], linestyle="dashed", color="red", alpha=0.3, label="Model")
        a1.legend()

        model.plot_loc_probs(output, a2)

        print(torch.max(torch.sigmoid(output)))

        plt.show()
        exit()



if __name__ == "__main__":
    main()
