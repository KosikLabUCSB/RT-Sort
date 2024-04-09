from src.model import ModelSpikeSorter
from torchsummary import summary


# ARCHITECTURE_PARAMS = {
#     "conv1_chans": 25,
#     "conv1_size": 3,
#     "conv1_stride": 1,
#     "conv1_pad": 0,
#
#     "pool1_size": 2,
#     "pool1_stride": None,
#     "pool1_pad": 0,
#
#     ###########################
#
#     "conv2_chans": 50,
#     "conv2_size": 5,
#     "conv2_stride": 1,
#     "conv2_pad": 0,
#
#     "pool2_size": 2,
#     "pool2_stride": None,
#     "pool2_pad": 0,
#
#     ###########################
#
#     "conv3_chans": 25,
#     "conv3_size": 3,
#     "conv3_stride": 1,
#     "conv3_pad": 0,
#
#     "pool3_size": 2,
#     "pool3_stride": None,
#     "pool3_pad": 0,
#
#     ###########################
#
#     "gpool": False,
#     "gpool_size": 3,
#     "gpool_stride": 1,
#     "gpool_pad": 0,
#
#     "linear1_out": 200,
# }
# model = ModelSpikeSorter(num_channels_in=1,
#                          sample_size=200, buffer_front_sample=20, buffer_end_sample=40,
#                          loc_prob_thresh=0.35, buffer_front_loc=0, buffer_end_loc=0,
#                          device="cuda",
#                          architecture_params=ARCHITECTURE_PARAMS)
model = ModelSpikeSorter.load("/data/MEAprojects/DLSpikeSorter/models/v0_3/2954")

summary(model, (model.num_channels_in, model.sample_size))



