"""Save model's outputs as .npy files for further testing"""

from src.model import ModelSpikeSorter
from src.data import RecordingCrossVal
from src import utils
import torch
import numpy as np

utils.random_seed(231)

MODEL_PATH = r"E:\KosikLab\Deep-Learning-for-Realtime-Spike-Sorting\data\models\v0_2\2950"
model = ModelSpikeSorter.load(MODEL_PATH)

recording_cross_val = RecordingCrossVal(sample_size=model.sample_size, front_buffer=model.buffer_front_sample, end_buffer=model.buffer_end_sample, num_spikes_probs=[100],
                                        thresh_amp=10, thresh_std=None,
                                        batch_size=1, num_workers=0)

for name, train, val in recording_cross_val:
    for i, (inputs, labels) in enumerate(val):
        if i != 1:  # Load a specify data sample (only works with batch_size=1)
            continue

        with torch.no_grad():
            outputs = model(inputs)
            preds = model.outputs_to_preds(outputs)
            model.plot_preds(inputs, outputs, preds, labels)

        np.save(r"E:\KosikLab\Deep-Learning-for-Realtime-Spike-Sorting\data\outputs.npy", outputs[0, :].cpu().numpy())
        exit()

    break





