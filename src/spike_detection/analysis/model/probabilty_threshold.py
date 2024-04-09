"""
Find which probability threshold for spike detection has the best results
"""

from src.model import ModelSpikeSorter
from src.data import RecordingCrossVal

model = ModelSpikeSorter.load(r"/data/MEAprojects/DLSpikeSorter/models/v0_4/2954").to("cuda")
recording_cross_val = RecordingCrossVal(sample_size=model.sample_size, front_buffer=model.buffer_front_sample, end_buffer=model.buffer_end_sample,
                                        num_wfs_probs=[0.6, 0.24, 0.12, 0.04],
                                        isi_wf_min=5, isi_wf_max=None,
                                        thresh_amp=3, thresh_std=0.6,
                                        batch_size=1, num_workers=0,
                                        mmap_mode="r",
                                        device="cuda")
# num_wfs_probs=[0.6, 0.24, 0.12, 0.04]
# for i, (rec, train, val) in enumerate(recording_cross_val):

rec, train, val = recording_cross_val["2954"]

print(f"Rec: {rec} - Train Num Samples: {len(train)} - Val Num Samples: {len(val)}\n")

model.tune_loc_prob_thresh(val)

perf = model.perf(val)
model.perf_report("\nVal", perf)
