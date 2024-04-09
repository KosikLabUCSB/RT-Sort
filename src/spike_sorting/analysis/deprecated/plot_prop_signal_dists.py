import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt


propagations = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/prop_signal/list_of_propagation_final.npy", allow_pickle=True)

latency = 0
for prop in propagations:
    latency = max(latency, *prop.latency)
print(latency)
exit()
dict_mat = {"propagations": [prop.to_numpy() for prop in propagations]}
savemat("/data/MEAprojects/dandi/000034/sub-mouse412804/prop_signal/list_of_propagation_final.mat", dict_mat)
exit()

num_electrodes = [len(prop) for prop in propagations]
bars = {}
for num in num_electrodes:
    if num not in bars:
        bars[num] = 1
    else:
        bars[num] += 1
plt.bar(list(bars.keys()), list(bars.values()))
plt.ylabel("Count")
plt.xlabel("Number of electrodes in propagation")
plt.show()

latencies = [lat for prop in propagations for lat in prop["latency"].values[1:]]  # [1:] to not include first electrode
bars = {}
for lat in latencies:
    if lat not in bars:
        bars[lat] = 1
    else:
        bars[lat] += 1
plt.bar(list(bars.keys()), list(bars.values()), width=0.05*0.8)
plt.ylabel("Count")
plt.xlabel("Latency with first electrode in propagation (ms)")
plt.show()
