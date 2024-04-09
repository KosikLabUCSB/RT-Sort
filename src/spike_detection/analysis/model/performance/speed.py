# TODO: Time does not include outputs_to_preds and subtracting mean from data

import sys
sys.path.append("/data/MEAprojects/DLSpikeSorter")
sys.path.append("/data/MEAprojects/DLSpikeSorter/src")

# Imports
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

from src.model import ModelSpikeSorter, UNet, ModelTuning
from src.data import MultiRecordingDataset
from src.utils import random_seed
from src.plot import set_dpi

import torch_tensorrt
import time
import numpy as np
import matplotlib.pyplot as plt


def get_model():
    # Load model
    MODEL_PATH = "/data/MEAprojects/DLSpikeSorter/test/model"
    model = ModelSpikeSorter.load(MODEL_PATH).model.conv  # Get barebone model
    # model.append(nn.Flatten())
    # model = ModelTuning(4, 50, "relu", 0, 0, 0, 0, 200)

    # Build model
    # Astro model scaled down to match 200 sample size
    # model = nn.Sequential(  # Input: (batch_size, num_channels_in=1, sample_size=200)
    #     nn.Conv1d(in_channels=1, out_channels=100, kernel_size=32, stride=2),  # (batch_size, 100, 85)
    #     nn.ReLU(),
    #
    #     nn.MaxPool1d(kernel_size=7, stride=4, padding=1),
    #
    #     nn.Conv1d(in_channels=100, out_channels=96, kernel_size=16, stride=1, padding=8),
    #     nn.ReLU(),
    #
    #     nn.MaxPool1d(kernel_size=6, stride=4, padding=1),
    #
    #     nn.Conv1d(in_channels=96, out_channels=96, kernel_size=16, stride=1, padding=8),
    #     nn.ReLU(),
    #
    #     nn.MaxPool1d(kernel_size=6, stride=4, padding=2),
    #
    #     nn.Flatten(),
    #     nn.Linear(192, 140),
    #
    # )
    # # CalTech BAR model
    # model = nn.Sequential(
    #     nn.Conv1d(in_channels=num_channels_in, out_channels=25, kernel_size=3, stride=1, padding=1),
    #     nn.ReLU(),
    #
    #     nn.Conv1d(in_channels=25, out_channels=50, kernel_size=5, stride=1, padding=0),
    #     nn.ReLU(),
    #
    #     nn.MaxPool1d(kernel_size=2, stride=1),
    #
    #     nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1),
    #     nn.ReLU(),
    #
    #     nn.MaxPool1d(kernel_size=2, stride=1),
    #
    #     nn.Flatten(),
    #     nn.Linear(9500, 50),
    #     nn.ReLU(),
    #
    #     nn.Linear(50, 140)
    # )

    # model = nn.Sequential(
    #     # nn.Conv1d(1, 200, 10),
    #     # nn.Conv1d(200, 200, 7),
    #     # nn.Conv1d(200, 200, 7),
    #     # nn.Conv1d(200, 200, 7),
    #     # 11.75 ms with 1020 batch size
    #
    #     # nn.Conv1d(1, 100, 10),
    #     # nn.Conv1d(100, 100, 7),
    #     # nn.Conv1d(100, 100, 7),
    #     # nn.Conv1d(100, 100, 7),
    #     # nn.Conv1d(100, 100, 7),
    #     # nn.Conv1d(100, 100, 7),
    #     # nn.Conv1d(100, 100, 7),
    #     # nn.Conv1d(100, 100, 7),
    #     # 7.78 ms with 1020 batch size
    #
    #     # nn.Conv1d(1, 100, 3),
    #     # nn.Conv1d(100, 100, 3),
    #     # nn.Conv1d(100, 100, 3),
    #     # nn.Conv1d(100, 100, 3),
    #     # nn.Conv1d(100, 100, 3),
    #     # nn.Conv1d(100, 100, 3),
    #     # nn.Conv1d(100, 100, 3),
    #     # nn.Conv1d(100, 100, 3),
    #     # # 4.62 ms with 1020 batch size
    #     # nn.Flatten(),
    #     # nn.Linear(100*184, 128),
    #     # nn.Linear(128, 140)
    #     # # 4.76 ms with 1020 batch size
    #
    #     # nn.Conv1d(1, 200, 5),
    #     # nn.Conv1d(200, 200, 3),
    #     # nn.Conv1d(200, 200, 3),
    #     # nn.Conv1d(200, 200, 3),
    #     # 6.25 ms with 1020 batch size
    #
    #     nn.Conv1d(1, 100, 5),
    #     nn.Conv1d(100, 100, 3),
    #     nn.Conv1d(100, 100, 3),
    #     nn.Conv1d(100, 100, 3),
    #     # 2.30 ms with 1020 batch size
    # )
    # model = UNet()

    # model = ModelTuning(7, 50, "relu", 0, 0, 0, 0, 200)

    model.to("cuda")
    model.eval()

    return model


def warmup(model, inputs, n_runs, verbose=True):
    # GPU needs to warmup

    if verbose:
        print("Warming up ...")
    with torch.no_grad():
        for _ in range(n_runs):
            model(inputs.to("cuda"))
            torch.cuda.synchronize()


def get_avg_speed(model, inputs, n_runs, verbose=True):
    # Get the average speed of model on n_runs

    if verbose:
        print("Start timing ...")

    speeds = []  # ms
    with torch.no_grad():
        for i in range(1, n_runs + 1):
            start_time = time.time()
            # model.outputs_to_preds(model(inputs.to("cuda")))
            model(inputs.to("cuda"))
            torch.cuda.synchronize()
            end_time = time.time()
            speed = (end_time - start_time) * 1000
            speeds.append(speed)
            if verbose and i % 10 == 0:
                print(f"Iteration {i}/{n_runs}, avg time: {speed:.2f} ms")
    return np.mean(speeds)


def main():
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1020, 2048]
    # BATCH_SIZES = [1020]
    DTYPE = torch.float16

    torch.backends.cudnn.benchmark = True

    # Get model
    model_og = get_model().to(DTYPE)
    in_channels = 1  # model_og.num_channels_in
    sample_size = 200  # 200  # model_og.sample_size
    front_buffer = 40  # model_og.buffer_front_sample
    end_buffer = 40  # model_og.buffer_end_sample

    # Get input data
    # dataset = MultiRecordingDataset.load_single(path_folder="/data/MEAprojects/DLSpikeSorter/2954",
    #                                             samples_per_waveform=100, front_buffer=front_buffer, end_buffer=end_buffer,
    #                                             num_wfs_probs=[0.6, 0.24, 0.12, 0.04], isi_wf_min=5, isi_wf_max=None,
    #                                             sample_size=sample_size,
    #                                             device="cpu", dtype=DTYPE,
    #                                             thresh_amp=3, thresh_std=0.6)
    random_seed(231)
    # inputs_all = torch.stack([dataset[i][0] for i in range(max(BATCH_SIZES))])

    # Start testing input sizes
    speed_avgs = []
    for size in BATCH_SIZES:
        # inputs = inputs_all[:size]

        # Compile model
        # model = model_og
        model = torch.jit.trace(model_og, [inputs.to("cuda")])
        model = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((size, in_channels, sample_size))], enabled_precisions={DTYPE})

        # Speed tests
        warmup(model, inputs, n_runs=50, verbose=False)
        speed_avg = get_avg_speed(model, inputs, n_runs=100, verbose=False)
        speed_avgs.append(speed_avg)

        print(f"{size} samples: {speed_avg:.2f} ms")

    # np.save("/data/MEAprojects/DLSpikeSorter/results/2023_vcsf/model_idk_speed.npy", np.vstack((BATCH_SIZES, speed_avgs)))
    plt.plot(BATCH_SIZES, speed_avgs)
    plt.scatter(BATCH_SIZES, speed_avgs)
    setup_plot()

    # Profiler
    # with torch.no_grad():
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #         with record_function("model inference"):
    #             model(inputs)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def plot_speeds(speed_paths: list, model_names: list):
    """
    Plot a scatter plot of model speed vs batch_size

    :param speed_paths:
        List of .npy file paths in which array[0] = batch_sizes and array[1] = speed
    :param model_names:
        List of str names for the previous speeds
    """

    for path, name in zip(speed_paths, model_names):
        batch_sizes, speeds = np.load(path)
        plt.plot(batch_sizes, speeds)
        plt.scatter(batch_sizes, speeds, label=name)

    setup_plot()


def setup_plot():
    # Setup plot title, label, axes, etc
    plt.title("Model Speed")
    plt.xlabel("Number of sampled electrodes")
    plt.ylabel("Computation time (ms)")
    plt.ylim(0)
    plt.xlim(0, 2070)

    # plt.axhline(1.5, color="gray", linestyle="dashed", alpha=0.7)
    # plt.axvline(1020, color="gray", linestyle="dashed", alpha=0.7)
    # plt.scatter(1020, 1.5, color="gray", label="Goal 2", alpha=0.7)

    plt.legend()
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    # main()

    # set_dpi(400)
    
    # plt.plot([1, 2, 3])
    # plt.show()
    
    plot_speeds(
        [
            # "/data/MEAprojects/DLSpikeSorter/models/v0_2/model_speed.npy",
            "/data/MEAprojects/DLSpikeSorter/models/v0_4_4/model_speed.npy",
            "/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/dl_models/240116/model_speed.npy"
            # "/data/MEAprojects/DLSpikeSorter/results/2023_vcsf/model_7_speed.npy",
         ],
        ["MEA model", "Neuropixels model"]
        # ["AR0", "AR4", "AR7"]
    )
    # plt.show()
    
    SAVE_NAME = "model_speeds"
    import pickle
    plt.savefig(f"/data/MEAprojects/RT-Sort/figures/230119_presentation/{SAVE_NAME}.png", format="png")
    with open(f'/data/MEAprojects/RT-Sort/figures/230119_presentation/{SAVE_NAME}.pickle', "wb") as file:
        pickle.dump(plt.gcf(), file)

    # print(np.load("/data/MEAprojects/DLSpikeSorter/results/2023_vcsf/model_7_speed.npy"))


