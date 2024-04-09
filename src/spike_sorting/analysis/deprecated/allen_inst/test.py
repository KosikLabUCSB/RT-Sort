from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache, EcephysSession
import numpy as np


# cache = EcephysProjectCache.from_warehouse(manifest="/data/MEAprojects/allen_inst/cache_manifest/manifest.json")
# session = cache.get_session_data(766640955)
# PROBE_ID = 773592320
# probes = cache.get_probes()
# channels = session.channels
# channels = channels[channels["probe_id"] == PROBE_ID]
# print(len(channels))
#
#
# import matplotlib.pyplot as plt
# for x,y in channels[["probe_horizontal_position", "probe_vertical_position"]].values:
#     plt.scatter(x, y)
# plt.show()
#
# print()


from spikeinterface.extractors import NwbRecordingExtractor
rec = NwbRecordingExtractor("/data/MEAprojects/dandi/000034/sub-mouse412804/sub-mouse412804_ecephys.nwb")

import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

fig, axs = plt.subplots(1)  # type: Figure, Axes
for x,y in rec.get_channel_locations():
    axs.scatter(x, y, marker="s", color="black", s=1)
plt.show()


print()
