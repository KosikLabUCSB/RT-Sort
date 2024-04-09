"""Sanity check for new sorted.npz files that contain the amplitudes and std in amplitude for each channel of each unit"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(r"C:\KosikLab")

total_wf = 0
removed_wf = 0
for rec in ROOT.iterdir():
    if not rec.name.isnumeric():
        continue

    rec = np.load(rec / "sorted.npz", allow_pickle=True)
    for i, unit in enumerate(rec["units"]):
        chan_max = int(unit["max_channel_si"])

        id = int(unit["unit_id"])
        peak_idx = unit["peak_ind"][chan_max]

        template = unit["template"]

        amplitude = unit["amplitudes"][chan_max] * 6.295
        std = unit["std_norms"][chan_max]
        # if amplitude < 20:
        #     print(f"{i}: Amplitude: {amplitude}")
        if std > 0.3:
            # print(f"{i}: Std: {std}")
            removed_wf += 1
        total_wf += 1
print(total_wf)
print(removed_wf)
print(total_wf - removed_wf)  # Should equal 173
