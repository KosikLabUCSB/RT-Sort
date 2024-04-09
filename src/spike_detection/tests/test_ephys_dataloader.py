from .. import custom_ephys_dataloader
import pandas as pd
import numpy as np

batch_uuid = "2022-06-30-e-UCSB-200123-2950-control"
experiment_num = "experiment1"  # Says it should be int in docstring, but must be str
sample_size = 10  # Must be same as 1st (0 indexing) column of csv file

"""
For attr_csv:
3 columns (offset, length, channels)
N rows - First row must be header. Header can be anything but must be something
       - Each following row represents a sample of the recording

offset is relative to sample extracted from recording and ∈ [0, length)
length should equal sample_size, both representing length of the sample to extract from recording
channels can be "all" to extract all channels or channel1/channel2/channel3/... (integers separated by "/" for extracting specific channels)
         must be at least 2 channels channel1/channel2 (can be easily fixed by making source code [channels])
         channels refer to channel indices in recording file, so they are channel IDs in spikeinterface and my code
"""
attr_csv = "2950_atr.csv"  # path to .csv file
start = 100000  # Frame to start taking data from in recording file

dataset = ephys_dataloader.EphysDataset(batch_uuid, experiment_num, sample_size, attr_csv, start)  # Data is unscaled
data = dataset[0]
