"""
Use gradient ascent to create a trace that causes the model to predict a spike or maximize location score
"""

from src.model import ModelSpikeSorter
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import torch
from src import plot

MODEL_PATH = r"/data/MEAprojects/DLSpikeSorter/models/v0_3/2954"
NUM_ITER = 50  # Number of iterations for gradient ascent
LR = 0.5  # Learning rate for gradient ascent
LOC_ASCEND = 100  # Location gradient ascent
TRACE_STD = 0  # Standard deviation of random trace (random normal distribution)


def main():
    plot.set_dpi(400)

    model = ModelSpikeSorter.load(MODEL_PATH)
    trace_og = torch.randn((1, 1, model.sample_size), device="cuda", dtype=torch.float32) * TRACE_STD
    trace = trace_og.clone().requires_grad_()
    idx_ascend = model.loc_to_logit(LOC_ASCEND)

    for i in range(NUM_ITER):
        trace.grad = None

        output = model(trace)
        output[0, idx_ascend].backward()
        with torch.no_grad():
            trace += LR * trace.grad

    with torch.no_grad():
        output = model(trace)
        pred = model.outputs_to_preds(output)[0]
        output = output[0, :]
        prob = torch.sigmoid(output[idx_ascend]) * 100
        trace = trace[0, 0, :].detach().cpu().numpy()

    fig, (a0, a1, a2) = plt.subplots(3, tight_layout=True, figsize=(6, 6))  # type: Axes

    plot.set_ticks((a0, a1, a2), trace)

    a0.plot(trace_og[0, 0, :].cpu().numpy())
    a0.set_title("Initial Input")

    a1.plot(trace)
    a1.set_title(f"Probability of spike: {prob:.2f}%")
    a1.axvline(LOC_ASCEND, linestyle="dashed", color="blue", alpha=0.5, label="Maximize")
    for loc in pred:
        a1.axvline(loc, linestyle="dashed", color="red", alpha=0.5, label="Model")
    a1.legend()

    model.plot_loc_probs(output, a2)

    plt.show()


if __name__ == "__main__":
    main()