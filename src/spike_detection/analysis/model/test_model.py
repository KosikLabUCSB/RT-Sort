from src.model import ModelSpikeSorter
from src.data import RecordingCrossVal
from src.utils import random_seed
from src.plot import set_dpi

set_dpi(400)
random_seed(123)
model = ModelSpikeSorter.load(r"E:\KosikLab\Deep-Learning-for-Realtime-Spike-Sorting\data\models\v0_2\2950")
recording_cross_val = RecordingCrossVal(sample_size=model.sample_size, front_buffer=model.buffer_front_sample, end_buffer=model.buffer_end_sample,
                                        num_wfs_probs=[0.6, 0.24, 0.12, 0.04], isi_wf_min=5, isi_wf_max=None,
                                        thresh_amp=3, thresh_std=0.6,
                                        batch_size=100, num_workers=0)

for i, (rec, train, val) in enumerate(recording_cross_val):
    print(f"Train Num Samples: {len(train.dataset)} - Val Num Samples: {len(val.dataset)}")
    perf = model.perf(val, plot_preds=("hist",), loc_buffer=10)
    model.perf_report("Val", *perf)
    break
