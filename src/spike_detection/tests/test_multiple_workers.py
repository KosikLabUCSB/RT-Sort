"""Figure out how to use multiple workers for PyTorch Dataloader"""

import torch
from torch.utils.data import Dataset, DataLoader
from src.data import WaveformDataset, MultiRecordingDataset, RecordingDataloader, RecordingCrossVal


class TestDataset(Dataset):
    def __init__(self, len):
        self._len = len

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return idx


def test_basic_dataloader():
    dataset = TestDataset(100)
    dataloader = DataLoader(dataset, batch_size=80, shuffle=True, num_workers=4)
    for batch in dataloader:
        print(batch)


def test_waveform_dataset():
    dataset = WaveformDataset(r"E:\KosikLab\Deep-Learning-for-Realtime-Spike-Sorting\data\2953\sorted.npz", 1.5)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
    for batch in dataloader:
        print(batch)


def test_multirecording_dataset():
    print("Getting dataset")
    dataset = MultiRecordingDataset(2, 20, 40, [100],
                                    [r"E:\KosikLab\Deep-Learning-for-Realtime-Spike-Sorting\data\2953\data.raw.h5"], 200,
                                    [r"E:\KosikLab\Deep-Learning-for-Realtime-Spike-Sorting\data\2953\sorted.npz"], 1.5, None,)
    print("Getting dataloader")
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=1)
    print("Starting loop")
    for batch in dataloader:
        print(batch)


def test_recording_cross_val():
    folds = RecordingCrossVal(200, 20, 40, [100],
                              1.5, None, num_workers=4, as_datasets=True)
    for name, train, val in folds:
        for batch in RecordingDataloader(train, batch_size=100, shuffle=True, num_workers=0):
            print(batch)
        exit()


def main():
    test_multirecording_dataset()


if __name__ == "__main__":
    main()

