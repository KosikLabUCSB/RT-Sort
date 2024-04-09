import comet_ml
import torch
import numpy as np
from src.data import RecordingCrossVal
from src.model import ModelSpikeSorter
from src.utils import random_seed
from torchsummary import summary


REC_CROSS_VAL_KWARGS = {
        "sample_size": 200,
        "front_buffer": 40,
        "end_buffer": 40,

        "num_wfs_probs": [0.6, 0.24, 0.12, 0.04],
        "isi_wf_min": 5,
        "isi_wf_max": None,
        "thresh_amp": 3,
        "thresh_std": 0.6,
        "samples_per_waveform": (2, 20),

        "data_root": "/data/MEAprojects/DLSpikeSorter",
        "mmap_mode": "r",
        "device": "cuda",
        "as_datasets": False,

        "num_workers": 0,
        "shuffle": True,
        "batch_size": 1
    }

MODEL_KWARGS = {
    "num_channels_in": 1,
    "sample_size": REC_CROSS_VAL_KWARGS["sample_size"],
    "buffer_front_sample": REC_CROSS_VAL_KWARGS["front_buffer"],
    "buffer_end_sample": REC_CROSS_VAL_KWARGS["end_buffer"],
    "loc_prob_thresh": 10,  # 0.00571428571429 * 100,
    "buffer_front_loc": 0,
    "buffer_end_loc": 0,
    "device": REC_CROSS_VAL_KWARGS["device"]
}

print("Using 'torch.backends.cudnn.benchmark = True'")
torch.backends.cudnn.benchmark = True


def tune_hyperparameters():
    EXPERIMENT_CONFIG = {
        "api_key": "max_api_key",
        "project_name": "spikesorter",
        "workspace": "max-workspace",

        "display_summary_level": 0,
        #"auto_param_logging": False,
        #"auto_metric_logging": False
    }
    EXP_NAME_BASE = "ModelTuning4.3_"  # val_f1_ModelTuning_Name (Ex: 94_ModelTuning_221228_130121_123810)

    OPTIMIZER_CONFIG = {
        "algorithm": "bayes",
        "spec": {
            "maxCombo": 0,
            "metric": "val_f1_score_final",
            "objective": "maximize",
        },

        "parameters": {
            # "architecture": {"type": "discrete", "values": [4]},
            # "num_channels": {"type": "discrete", "values": [25, 50, 100, 150]},
            # "relu": {"type": "categorical", "values": ["relu", "prelu"]},
            # "add_conv": {"type": "integer", "scaling_type": "uniform", "min": 0, "max": 6},  # Additional 3/1 conv layers
            # "bottleneck": {"type": "discrete", "values": [0, 1]},  # Whether to use a 1/1 conv layer for last layer
            # "noise": {"type": "discrete", "values": [0, 0.5]},  # Whether to use noise in last layer output locations. 0.5=use fully connected layer instead of RMS for noise
            # "filter": {"type": "discrete", "values": [0, 1]},  # Whether to use bandpass filter
            # "init": {"type": "categorical", "values": ["kaiming", "xavier"]},

            "lr": {"type": "float", "scaling_type": "uniform", "min": 1e-5, "max": 8e-4},
            "lr_factor": {"type": "float", "scaling_type": "uniform", "min": 0.2, "max": 0.8},
            "momentum": {"type": "float", "scaling_type": "uniform", "min": 0.75, "max": 0.86},  #  [0.85, 0.95]
            "optim": {"type": "categorical", "values": ["momentum", "nesterov"]},
            # "num_wf_locs": {"type": "categorical", "values": [  # 1_wf_prob%2_wf_prob%3_wf_prob ...
            #     "60%24%12%4%",
            #     "5%15%30%"
            # ]},
        }
    }
    # P = {}  # OPTIMIZER_CONFIG["parameters"]
    # for i in range(1, 4):
    #     P[f"conv{i}_chans"] = (32, 50, 64, 100, 128, 256, 512)
    #     P[f"conv{i}_size"] = (16,)
    #     P[f"conv{i}_stride"] = (1,)  # (1, 3, 5)
    #     P[f"conv{i}_pad"] = (0,)  # (0, 2, 4)
    #
    #     P[f"pool{i}_size"] = (0,)  # (0, 2, 4)
    #     P[f"pool{i}_stride"] = (-1,)  # (-1, 1, 2)  # -1 = None
    #     P[f"pool{i}_pad"] = (0,)
    #
    # ARCHITECTURE_PARAMS_KEYS = P.keys()
    # for p in P:
    #     P[p] = {"type": "discrete", "values": P[p]}
    opt = comet_ml.Optimizer(config=OPTIMIZER_CONFIG)

    for i_exp, exp in enumerate(opt.get_experiments(**EXPERIMENT_CONFIG)):
        architecture = 4  # exp.get_parameter("architecture")
        num_channels = 50  # exp.get_parameter("num_channels")
        relu = "relu"  # exp.get_parameter("relu")
        add_conv = 0  # exp.get_parameter("add_conv")
        bottleneck = 0  # exp.get_parameter("bottleneck")
        noise = 0  # exp.get_parameter("noise")
        filter = 0  # exp.get_parameter("filter")
        init = "xavier"  # exp.get_parameter("init")

        lr = exp.get_parameter("lr")
        lr_factor = exp.get_parameter("lr_factor")
        momentum = exp.get_parameter("momentum")
        optim = exp.get_parameter("optim")

        if architecture == 7:
            REC_CROSS_VAL_KWARGS["sample_size"] = 204
            REC_CROSS_VAL_KWARGS["front_buffer"] = 44
            REC_CROSS_VAL_KWARGS["end_buffer"] = 44
        elif architecture >= 4:
            REC_CROSS_VAL_KWARGS["sample_size"] = 200
            REC_CROSS_VAL_KWARGS["front_buffer"] = 40
            REC_CROSS_VAL_KWARGS["end_buffer"] = 40
        else:
            REC_CROSS_VAL_KWARGS["sample_size"] = 200
            REC_CROSS_VAL_KWARGS["front_buffer"] = 20
            REC_CROSS_VAL_KWARGS["end_buffer"] = 40

        MODEL_KWARGS = {
            "num_channels_in": 1,
            "sample_size": REC_CROSS_VAL_KWARGS["sample_size"],
            "buffer_front_sample": REC_CROSS_VAL_KWARGS["front_buffer"],
            "buffer_end_sample": REC_CROSS_VAL_KWARGS["end_buffer"],
            "loc_prob_thresh": 10,  # 0.005714285714x29 * 100,
            "buffer_front_loc": 0,
            "buffer_end_loc": 0,
            "device": REC_CROSS_VAL_KWARGS["device"]
        }

        rec_cross_val = RecordingCrossVal(**REC_CROSS_VAL_KWARGS)
        val_f1_score_final_total = 0
        for rec, train, val in rec_cross_val:
            # if rec != "2954" and rec != "2953": continue
            rec_cross_val.summary(rec)
            model = ModelSpikeSorter(**MODEL_KWARGS,
                                     architecture_params=(architecture, num_channels,
                                                          relu, add_conv, bottleneck, noise, filter))
            model.init_weights_and_biases(init, prelu_init=0 if relu == "relu" else 0.25)
            model.model.init_final_bias(
                MODEL_KWARGS["sample_size"] - MODEL_KWARGS["buffer_front_sample"] - MODEL_KWARGS["buffer_end_sample"],
                REC_CROSS_VAL_KWARGS["num_wfs_probs"])
            summary(model, (model.num_channels_in, model.sample_size))
            train_perf, val_perf, val_f1_score_final = model.fit(train, val, optim,
                                                            num_epochs=200, epoch_patience=10,
                                                            lr=lr, lr_patience=5, lr_factor=lr_factor,
                                                            tune_thresh_every=10,
                                                            momentum=momentum)
            model.save(f"/data/MEAprojects/DLSpikeSorter/models/v0_4_4/{rec}")
            val_f1_score_final_total += val_f1_score_final

        val_f1_score_final_avg = val_f1_score_final_total / len(rec_cross_val)
        exp.log_metric("val_f1_score_final", val_f1_score_final_avg)
        exp.set_name(EXP_NAME_BASE + f"{val_f1_score_final_avg:.3f}")

        # for dataset, perfs in zip(("train", "val"), (train_perf, val_perf)):
        #     for epoch, perf in enumerate(perfs):
        #         for metric, value in zip(("loss", "wf_detected", "accuracy", "recall", "precision", "f1_score", "loc_mad_frames", "loc_mad_ms"), perf):
        #             exp.log_metric(f"{dataset}_{metric}", value, epoch=epoch)

        # model_name = model.save(f"/data/MEAprojects/DLSpikeSorter/models/v0_4/{VAL_REC}")
        # exp.set_name(EXP_NAME_BASE + model_name)


def train_model():
    rec_cross_val = RecordingCrossVal(**REC_CROSS_VAL_KWARGS)

    # model = ModelSpikeSorter(**MODEL_KWARGS, architecture_params=(4, 50, "relu", 0, 0, 0, 0))
    # # model = ModelSpikeSorter.load("/data/MEAprojects/DLSpikeSorter/models/v0_4/2954")
    # # for param in model.parameters():
    # #     if param.shape[0] != 140:  # Don't freeze last layer (weights + biases)
    # #         param.requires_grad = False
    #
    # model.init_weights_and_biases("xavier", prelu_init=0)
    # model.model.init_final_bias(model.num_output_locs, REC_CROSS_VAL_KWARGS["num_wfs_probs"])
    #
    # from torchsummary import summary
    # summary(model, (model.num_channels_in, model.sample_size))

    for rec, train, val in rec_cross_val:
        rec_cross_val.summary(rec)

        model = ModelSpikeSorter(**MODEL_KWARGS, architecture_params=(4, 50, "relu", 0, 0, 0, 0))
        model.init_weights_and_biases("xavier", prelu_init=0)
        model.model.init_final_bias(model.num_output_locs, REC_CROSS_VAL_KWARGS["num_wfs_probs"])

        model.fit(train, val, optim="momentum",
                  num_epochs=200, epoch_patience=10, epoch_thresh=0.01,
                  lr=7.76e-4, momentum=0.85, lr_patience=5, lr_factor=0.4,
                  tune_thresh_every=10)
        model.save("/data/MEAprojects/DLSpikeSorter/models/v0_4_4/" + rec)


if __name__ == "__main__":
    tune_hyperparameters()
    # train_model()
