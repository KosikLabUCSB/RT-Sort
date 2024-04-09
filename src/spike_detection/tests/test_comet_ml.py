import comet_ml
import torch
import os
import numpy as np


class Model:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def loss(self, x, y):
        y_hat = self.weight * x + self.bias
        return torch.sum(torch.abs(y - y_hat))


def main():
    os.environ["COMET_GIT_DIRECTORY"] = "/data/MEAprojects/DLSpikeSorter"

    INPUTS = torch.tensor([
        [0],
        [1],
        [2]
    ], dtype=torch.float32)

    OPTIMIZER_CONFIG = {
        "algorithm": "bayes",
        "spec": {
            "maxCombo": 0,
            "metric": "loss",
            "objective": "minimize",
        },
        "parameters": {
            "weight": {"type": "float", "scaling_type": "uniform", "min":-2, "max":2},
        },
        "trials": 1,
    }

    EXPERIMENT_CONFIG = {
        "api_key": "max_api_key",
        "project_name": "spikesorter",
        "workspace": "max-workspace",
        "auto_metric_logging": False,

        "display_summary_level": 0,
    }

    # opt = comet_ml.Optimizer(OPTIMIZER_CONFIG)
    # for i, exp in enumerate(opt.get_experiments(**EXPERIMENT_CONFIG)):
    #     print(f"#"*100)
    #     print(f"Iter: {i+1}")
    #
    #     weight = exp.get_parameter("weight")
    #     bias = 0
    #     model = Model(weight, bias)
    #     loss = model.loss(INPUTS, INPUTS)
    #     exp.log_metric("loss", loss)
    #     print(f"Iter {i+1}, weight: {weight}, loss: {loss}")

    exp = comet_ml.Experiment(**EXPERIMENT_CONFIG)
    train_perf = np.load("/data/MEAprojects/DLSpikeSorter/models/v0_3/2954/221125_235355/log/train_perf.npy")
    val_perf = np.load("/data/MEAprojects/DLSpikeSorter/models/v0_3/2954/221125_235355/log/val_perf.npy")
    for dataset, perfs in zip(("train", "val"), (train_perf, val_perf)):
        for epoch, perf in enumerate(perfs):
            for metric, value in zip(("loss", "wf_detected", "accuracy", "recall", "precision", "f1_score", "loc_mad_frames", "loc_mad_ms"), perf):
                exp.log_metric(f"{dataset}_{metric}", value, epoch=epoch)


if __name__ == "__main__":
    main()


