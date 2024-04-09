"""Average the model's performance across all holdout recordings"""

# Imports
import numpy as np
from pathlib import Path
from src.model import ModelSpikeSorter
from src.utils import random_seed, confusion_stats
from src import data, plot
import torch

# Constants and setup
DATA_PATHS = Path(r"/data/MEAprojects/DLSpikeSorter")
# Path must contain a list of folders where trained models are stored. The name of the folder is the corresponding holdout recording
MODEL_PATHS = r"/data/MEAprojects/DLSpikeSorter/models/v0_4_4"
PERF_CSV_PATH = r"/data/MEAprojects/DLSpikeSorter/models/v0_4_4/perf.csv"  # Path to save csv file with performances
PERF_CSV_HEADER = "Test Recording,Number of Samples,Loss,Accuracy (%),Recall (%),Precision (%),F1 Score (%),MAD of Location (ms)\n"  # Header of csv with performances
PERF_CSV_FORMAT = "{},{},{:.3f},{:.1f},{:.1f},{:.1f},{:.1f},{:.4f}\n"  # Format of saving perf (i.e. how many decimal places for each metric)
# plot.set_dpi(400)
SAMPLES_PER_WAVEFORM = 20
NUM_WFS_PROBS = [0.6, 0.24, 0.12, 0.04]
ISI_WF_MIN = 5
ISI_WF_MAX = None
THRESH_AMP = 3
THRESH_STD = 0.6
DTYPE = torch.float16


# Get total and individual perf stats across all models/recording holdouts
recording_names = []
recording_num_samples = []
recording_perfs = []

loc_deviations_all = []
for model_path in sorted([path for path in Path(MODEL_PATHS).iterdir()]):  # Iterate through each individual model/recording
    random_seed(231)

    # Get recording
    rec = model_path.name
    if not rec.isnumeric():
        continue

    # Get model
    model = ModelSpikeSorter.load(model_path).to(DTYPE)

    # Get data
    dataset = data.MultiRecordingDataset.load_single(path_folder=DATA_PATHS / rec,
                                                     samples_per_waveform=SAMPLES_PER_WAVEFORM, front_buffer=model.buffer_front_sample, end_buffer=model.buffer_end_sample,
                                                     num_wfs_probs=NUM_WFS_PROBS, isi_wf_min=ISI_WF_MIN, isi_wf_max=ISI_WF_MAX,
                                                     sample_size=model.sample_size,
                                                     thresh_amp=THRESH_AMP, thresh_std=THRESH_STD,
                                                     dtype=DTYPE)
    dataloader = data.RecordingDataloader(dataset, batch_size=10000)
    num_samples = len(dataset)
    print(f"{rec}: {num_samples} samples")

    # Get performance
    perf = model.perf(dataloader)
    # Report performance
    model.perf_report(rec, perf)

    # Cache performance
    recording_names.append(rec)
    recording_num_samples.append(num_samples)
    recording_perfs.append(perf)

# Report average of performances
average_perf = np.array(recording_perfs).mean(axis=0)
ModelSpikeSorter.perf_report("\nMean", average_perf)

# Report std of performances
std_perf = np.array(recording_perfs).std(axis=0)
ModelSpikeSorter.perf_report(" STD", std_perf)

# Save individual and average performance as csv (open in Excel then copy and paste into Google Slides)
recording_names.append('Mean')
recording_num_samples.append(np.around(np.mean(recording_num_samples), 1))
recording_perfs.append(average_perf)

recording_names.append('STD')
recording_num_samples.append(np.around(np.std(recording_num_samples), 1))
recording_perfs.append(std_perf)

# Only store relevant information in csv (loss, accuracy, recall, precision, f1, loc MAD in ms)
recording_perfs = np.array(recording_perfs)[:, [0, 2, 3, 4, 5, 7]]

with open(PERF_CSV_PATH, "w") as csv:
    csv.write(PERF_CSV_HEADER)
    for name, num_samples, perf in zip(recording_names, recording_num_samples, recording_perfs):
        csv.write(PERF_CSV_FORMAT.format(name, num_samples, *perf))

# Plot histograms with data from all recordings
# plot.plot_hist_loc_mad(loc_deviations_all)
# plot.plot_hist_percent_abs_error(alpha_percent_abs_errors_all)

