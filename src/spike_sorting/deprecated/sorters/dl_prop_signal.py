import json

from src.sorters.base import *
from src.prop_signal import v1_2


class DLPropSignal(SpikeSorter):
    def __init__(self, path, recording, name):
        self.path = Path(path)
        self.props = np.load(str(self.path / "propagations.npy"), allow_pickle=True)

        super().__init__(recording, name)

    def get_spike_times(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass






