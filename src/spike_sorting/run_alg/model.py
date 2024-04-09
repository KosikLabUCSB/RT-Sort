import torch
from torch import nn
import torch_tensorrt
import numpy as np
from scipy.signal import find_peaks
# from src import utils, plot, data
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# class InceptionBlock(nn.Module):
#     def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
#         super(InceptionBlock, self).__init__()
#
#         self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
#         self.branch2 = nn.Sequential(
#             ConvBlock(in_channels, red_3x3, kernel_size=1),
#             ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
#         )
#         self.branch3 = nn.Sequential(
#             ConvBlock(in_channels, red_5x5, kernel_size=1),
#             ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2)
#         )
#         self.branch4 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#             ConvBlock(in_channels, out_1x1pool, kernel_size=1)
#         )
#
#     def forward(self, x):
#         return torch.cat((self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)), 1)


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, **conv_kwargs):
#         super(ConvBlock, self).__init__()
#         self.relu = nn.PReLU()
#         self.conv = nn.Conv1d(in_channels, out_channels, **conv_kwargs)
#
#     def forward(self, x):
#         return self.relu(self.conv(x))


class ModelSpikeSorter(nn.Module):
    """DL model for spike sorting"""

    # Performance report when multiple waveforms can appear per sample
    _perf_report = "{}: Loss: {:.3f} | WF Detected: {:.1f}% | Accuracy: {:.1f}% | Recall: {:.1f}% | Precision: {:.1f}% | F1 Score: {:.1f}% | Loc MAD: {:.2f} frames = {:.4f} ms"
    compiled_name = "compiled.ts"

    def __init__(self, num_channels_in: int,
                 sample_size: int, buffer_front_sample: int, buffer_end_sample: int,
                 loc_prob_thresh: float = 35, buffer_front_loc: int = 0, buffer_end_loc: int = 0,
                 input_scale = 0.01,
                 device: str = "cuda", dtype=torch.float16,
                 architecture_params=None):
        """
        :param num_channels_in: int
            Number of channels int inputs

        :param sample_size: int
            Number of frames in inputs
        :param buffer_front_sample: int
            Model assumes all spikes are in [buffer_front_sample, sample_size - buffer_end_sample)
        :param buffer_end_sample: int
            Model assumes all spikes are in [buffer_front_sample, sample_size - buffer_end_sample)

        :param loc_prob_thresh: float
            If any frame has a probability of a spike occurring >= loc_prob_thresh (percent), the model will predict a spike occurred
        :param buffer_front_loc: int
            Model will predict the probability of a spike occurring in [buffer_front_sample - buffer_front_loc, sample_size - buffer_end_sample + buffer_end_loc)
         :param buffer_end_loc: int
            Model will predict the probability of a spike occurring in [buffer_front_sample - buffer_front_loc, sample_size - buffer_end_sample + buffer_end_loc)

        :param input_scale:
            Multiply input by this factor after subtracting mean

        :param device: str
            Device to run model ("cpu" for CPU and "cuda:0" for GPU)
        :param architecture_params
        """
        super(ModelSpikeSorter, self).__init__()

        # Cache init args
        self.num_channels_in = num_channels_in

        self.sample_size = sample_size
        self.buffer_front_sample = buffer_front_sample
        self.buffer_end_sample = buffer_end_sample

        # loc_prob_thresh needs to be upscaled from (0, 100) to (-inf, inf) since model's outputs are logits (no sigmoid)
        self.loc_prob_thresh_logit = 0
        self.set_loc_prob_thresh(loc_prob_thresh)

        self.buffer_front_loc = buffer_front_loc
        self.buffer_end_loc = buffer_end_loc

        # Cache for plotting localization
        self.loc_first_frame = self.buffer_front_sample - self.buffer_front_loc  # First frame in input that has a probability score for localization
        self.loc_last_frame = sample_size - buffer_end_sample + buffer_end_loc - 1  # Last frame in input that has a probability score for localization

        # Number of locations where model predicts a spike
        assert buffer_front_loc == buffer_end_loc == 0, "num_output_locs may not be implemented correctly if these are not equal to 0"
        self.num_output_locs = (sample_size - buffer_end_sample + buffer_end_loc) - (buffer_front_sample - buffer_front_loc)

        # region David's latest model on his github
        # conv0_kwargs = {"in_channels": num_channels_in, "out_channels": 100, "kernel_size": 30, "stride": 2, "padding": "valid"}
        # conv1_kwargs = {"in_channels": conv0_kwargs["out_channels"], "out_channels": 200, "kernel_size": 10, "stride": 2, "padding": "valid"}
        # conv2_kwargs = {"in_channels": conv1_kwargs["out_channels"], "out_channels": 300, "kernel_size": 10, "stride": 2, "padding": "valid"}
        # conv3_kwargs = {"in_channels": conv2_kwargs["out_channels"], "out_channels": 300, "kernel_size": 6, "stride": 2, "padding": "valid"}
        # conv_layer_kwargs = [conv0_kwargs, conv1_kwargs, conv2_kwargs, conv3_kwargs]
        # conv_layers = []
        # for kwargs in conv_layer_kwargs:
        #     conv_layers.extend([
        #         ModelSpikeSorter.get_same_padding(kwargs),
        #         nn.Conv1d(**kwargs),
        #         nn.ReLU(),
        #     ])
        # conv = nn.Sequential(*conv_layers)
        # linear = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(int(np.prod(ModelSpikeSorter.get_output_shape(conv, (num_channels_in, sample_size)))), 300),
        #     nn.ReLU(),
        # )
        # layer_output = nn.Linear(300, num_output_locs)
        # model = nn.Sequential(conv, linear, layer_output)
        # endregion

        # region David's model in his astronomy paper
        # Astro model scaled down to match 200 sample size
        # model = nn.Sequential(  # Input: (batch_size, num_channels_in=1, sample_size=200)
        #     nn.Conv1d(in_channels=num_channels_in, out_channels=100, kernel_size=32, stride=2),  # (batch_size, 100, 85)
        #     nn.ReLU(),
        #
        #     nn.MaxPool1d(kernel_size=7, stride=4, padding=1),
        #
        #     nn.Conv1d(in_channels=100, out_channels=96, kernel_size=16, stride=1, padding=8),
        #     nn.ReLU(),
        #
        #     nn.MaxPool1d(kernel_size=6, stride=4, padding=1),
        #
        #     nn.Conv1d(in_channels=96, out_channels=96, kernel_size=16, stride=1, padding=8),
        #     nn.ReLU(),
        #
        #     nn.MaxPool1d(kernel_size=6, stride=4, padding=2),
        #
        #     nn.Flatten(),
        #     nn.Linear(192, 350),
        #     nn.Linear(350, num_outputs_locs)
        # )
        # endregion

        # region Caltech BAR model
        # model = nn.Sequential(
        #     nn.Conv1d(in_channels=num_channels_in, out_channels=25, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(num_features=25),
        #
        #     nn.Conv1d(in_channels=25, out_channels=50, kernel_size=5, stride=1, padding=0),
        #     nn.ReLU(),
        #
        #     nn.MaxPool1d(kernel_size=2, stride=1),
        #     nn.BatchNorm1d(num_features=50),
        #
        #     nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1),
        #     nn.ReLU(),
        #
        #     nn.MaxPool1d(kernel_size=2, stride=1),
        #     nn.BatchNorm1d(50),
        #
        #     nn.Flatten(),
        #     nn.Linear(9500, 50),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(50),
        #
        #     nn.Linear(50, 140)
        # )
        # endregion

        # region Inception layers
        # model = nn.Sequential(
        #     InceptionBlock(1, 64, 96, 128, 16, 32, 32),
        #     nn.Flatten()
        # )
        # out_shape = ModelSpikeSorter.get_output_shape(layer=model, input_shape=(1, num_channels_in, sample_size))
        # model.append(nn.Linear(out_shape[1], 350))
        # model.append(nn.ReLU())
        # model.append(nn.Linear(350, num_output_locs))
        # endregion

        # region Tuning model 1
        # PRELU_INIT = 0.25
        #
        # self.architecture_params = architecture_params
        # p = architecture_params
        # """
        # chans = Num filters
        # size = Size of filters
        # stride = stride
        # pad = zero padding
        # """
        # model = nn.Sequential()
        # in_channels = num_channels_in
        # i = 1
        # while f"conv{i}_chans" in p:
        #     model.append(nn.Conv1d(in_channels=in_channels, out_channels=p[f"conv{i}_chans"], kernel_size=p[f"conv{i}_size"],
        #                            stride=p[f"conv{i}_stride"], padding=p[f"conv{i}_pad"]))
        #     # model.append(nn.ReLU())
        #     # model.append(nn.PReLU())
        #     model.append(nn.PReLU(num_parameters=p[f"conv{i}_chans"], init=PRELU_INIT))
        #     # layers.append(nn.SELU())
        #     if p[f"pool{i}_size"] > 0:
        #         stride = p[f"pool{i}_stride"]
        #         stride = stride if stride != -1 else None
        #         model.append(nn.MaxPool1d(kernel_size=p[f"pool{i}_size"], stride=stride, padding=p[f"pool{i}_pad"]))
        #     # layers.append(nn.BatchNorm1d(p[f"conv{i}_chans"]))
        #     # layers.append(nn.InstanceNorm1d(p[f"conv{i}_chans"]))
        #     in_channels = p[f"conv{i}_chans"]
        #     i += 1
        #
        # model.append(nn.Conv1d(in_channels=in_channels, out_channels=1,
        #                        kernel_size=p["conv_last_size"], stride=p["conv_last_stride"], padding=p["conv_last_pad"]))
        # # model.append(nn.Flatten())
        #
        # model.append(nn.PReLU(num_parameters=1, init=PRELU_INIT))
        # self.linear = nn.Conv1d(2, 1, 1)
        # self.flatten = nn.Flatten()

        # endregion

        # region Tuning model 2
        self.architecture_params = architecture_params
        if architecture_params is not None:
            model = ModelTuning(*architecture_params)
        else:
            model = RMSThresh(buffer_front=buffer_front_sample, buffer_end=buffer_end_sample)
        # endregion

        self.model = model

        self.input_scale = input_scale

        # Set device
        self.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

        # # Initialize weights and biases
        # self.init_weights_and_biases(0.25)

        self.logs = {}  # {file_name: contents}

        # Loss function
        self.loss_localize = nn.BCEWithLogitsLoss(reduction='none')

        self.path = None

    def init_weights_and_biases(self, method: str, prelu_init=0.25):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(module.weight, a=prelu_init, nonlinearity="leaky_relu")
                elif method == "xavier":
                    nn.init.xavier_normal_(module.weight)
                else:
                    raise ValueError(f"'{method}' is not a valid argument for parameter 'method'")
                nn.init.zeros_(module.bias)

    def init_final_bias(self, num_wfs_probs: list):
        """
        Initialize bias of the final layer based on the waveform probabilities of training dataset
        (assumes 50% of samples contain no waveform)

        :param num_wfs_probs:
            If there is at least 1 waveform in the sample, then the probability of i additional waveforms occurring
            in the sample is num_wfs_probs[i]
        """
        # Get the last weight layer
        # last_weight_layer = [module for module in self.modules() if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear)][-1]
        last_weight_layer = self.linear

        # Get the probability of a waveform occurring at a location probability
        exp_prob = 0
        for i, prob in enumerate(num_wfs_probs):
            exp_prob += prob * (i + 1)
        exp_prob *= 0.5 * 1/self.num_output_locs  # 50% chance of a waveform appearing at all, and 1/num_output_locs for waveform appearing at a output location
        # torch.sigmoid(bias) = exp_prob
        nn.init.constant_(last_weight_layer.bias, torch.logit(torch.tensor(exp_prob)).item())

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    def forward(self, x):
        # Normalize x --> Now done by dataloader
        # x = (x - torch.mean(x, dim=2, keepdim=True))  # / torch.std(x, dim=2, keepdim=True)

        # x = x / torch.std(x, dim=2, keepdim=True)
        return self.model(x * self.input_scale)  # self.model(x*self.input_scale)

        rms = torch.sqrt(torch.mean(torch.square(x), dim=(1, 2), keepdim=True))  # - 1.3
        x = torch.cat([self.model(x), rms.repeat(1, 1, self.num_output_locs)], dim=1)

        return self.flatten(self.linear(x))

    def loss(self, outputs, num_wfs, wf_locs):
        # num_wfs and wf_locs are the labels

        # Ind array containing which samples have wf
        wf_samples = num_wfs.type(torch.bool)

        # Localization loss
        # outputs_locs = outputs[wf_samples, self.idx_loc]
        # wf_logits = torch.clamp_min(self.loc_to_logit(wf_locs[wf_samples, :]), -1)
        # n_wf_samples, n_loc_logits = outputs_locs.shape
        # labels_loc = torch.zeros(n_wf_samples, n_loc_logits + 1, dtype=torch.float32, device=self.device)
        # row_ind = np.repeat(np.arange(n_wf_samples), wf_logits.shape[1])
        # labels_loc[row_ind, wf_logits.to(torch.long).flatten()] = 1
        # localize = self.loss_localize(outputs_locs, labels_loc[:, :-1])  # Only trains on samples with waveform

        # Sigmoid for probability instead of softmax
        wf_samples_ind = torch.nonzero(wf_samples).flatten()
        wf_logits = torch.clamp_min(self.loc_to_logit(wf_locs[wf_samples, :]), -1)
        labels_loc = torch.zeros(len(outputs), outputs.shape[1] + 1, dtype=torch.float32, device=outputs.device)
        wf_row_ind = np.repeat(wf_samples_ind.cpu(), wf_logits.shape[1])
        wf_col_ind = wf_logits.to(torch.long).flatten()
        labels_loc[wf_row_ind, wf_col_ind] = 1

        localize = self.loss_localize(outputs, labels_loc[:, :-1])  # Train on all samples

        # Only train on samples with wf
        # localize = self.loss_localize(outputs_locs[wf_samples_ind], labels_loc[wf_samples_ind, :-1])

        # num_wfs = torch.sum(num_wfs)
        # num_no_wfs = torch.numel(outputs) - num_wfs
        # wfs_multiplier = num_no_wfs / num_wfs / 10  # Multiply this by losses caused by a waveform location since there are many more locations without waveforms than with
        # localize[wf_row_ind, wf_col_ind] *= 50

        localize = torch.mean(torch.sum(localize, dim=1))  # Loss pretraining = -ln(0.5) * num_logits (number of neurons in output layer)
        # localize = torch.mean(localize)  # Loss pretraining: -ln(0.5) = 0.697

        # When there is no waveform in sample, correct probability is 1/num_possible_frames (equal probabilities across all frames)
        # outputs_localize = outputs[:, self.idx_loc]
        # labels_localize = torch.full_like(outputs_localize, 1 / outputs_localize.shape[1], dtype=torch.float32)
        # labels_localize[wf_samples, :] = torch.nn.functional.one_hot(self.loc_to_logit(labels[wf_samples, 1]), labels_localize.shape[1]).to(torch.float32)
        # localize = self.loss_localize(outputs_localize, labels_localize)

        return localize

    def train_epoch(self, dataloader, optim):
        self.train(True)
        # print("start")

        # utils.random_seed(231, silent=True)
        for inputs, num_wfs, wf_locs, wf_alphas in dataloader:
            # print("a")
            for param in self.parameters():
                param.grad = None

            outputs = self(inputs)
            # print("b")

            # Autograd
            self.loss(outputs, num_wfs, wf_locs).backward()
            # print("c")

            # Manually calculate gradient to push all probabilities to 0 - doesn't work
            # wf_samples = labels[:, 0].type(torch.bool)
            # grads = torch.zeros_like(outputs)
            # grads[:, self.idx_loc] = torch.softmax(outputs.detach()[:, self.idx_loc], dim=1)
            # correct_locs = self.loc_to_logit(labels[wf_samples, 1])
            # grads[wf_samples, 1+correct_locs] -= 1
            # grads /= grads.shape[0]
            # outputs.backward(grads)

            optim.step()
            # print("d")
        self.train(False)

    def fit(self, dataloader_train, dataloader_val, optim="adam",
            num_epochs=100, epoch_patience=10, epoch_thresh=0.5,
            lr=3e-4, momentum=0.9, lr_patience=5, lr_factor=0.1,
            tune_thresh_every=10, save_best=True):
        """
        Fit self to dataloader_train

        :param dataloader_train:
        :param dataloader_val:
        :param optim: str
            ("adam", "momentum", "nesterov")
        :param num_epochs:
        :param epoch_patience: int
            If not None and If loss does not decrease after epoch_patience epochs, then stop training
        :param epoch_thresh: float
            Loss must decrease by at least epoch_thresh to reset patience
        :param lr:
            Learning rate
        :param momentum:
            Momentum for SGD
        :param lr_patience: int
            If not None and If loss does not decrease after lr_patience, then lower learning rate by lr_factor
        :param lr_factor:
            Multiplicative factor to reduce learning rate
        :param tune_thresh_every:
            If not None, tune loc_prob_thresh every tune_thresh_every (int) epochs
        :param save_best:
            If True, save model weights that give best loss (new best has to be less than old best - epoch_thresh) 
                     and reset to this after training ends
        """
        train_start = time.time()

        assert optim in {"adam", "momentum", "nesterov"}
        if optim == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=lr)
        elif optim == 'momentum':
            optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, nesterov=False)
        elif optim == 'nesterov':
            optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, nesterov=True)
        else:
            raise ValueError(f"'{optim}' is not a valid argument for parameter 'optim'")

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode="min", factor=lr_factor, patience=lr_patience-1, verbose=True)
        # torch.optim.lr_scheduler.ExponentialLR(optim, 0.95, verbose=True)

        # Get performance before any training
        train_log = f"""Train: {len(dataloader_train.dataset)} samples
Valid: {len(dataloader_val.dataset)} samples
Num wfs probs: {dataloader_train.num_wfs_probs}

Before Training"""
        print(train_log)
        train_perf_all = [self.perf(dataloader_train)]
        train_log += "\n" + self.perf_report("Train", train_perf_all[0])
        val_perf_all = [self.perf(dataloader_val)]
        train_log += "\n" + self.perf_report("Valid", val_perf_all[0])

        best_loss = val_perf_all[0][0]
        epoch_patience_counter = 0  # Number of epochs since best loss 
        if save_best:
            best_weights = self.state_dict()

        # Start training!
        for epoch in range(1, num_epochs + 1):
            epoch_formatted = f"\nEpoch: {epoch}/{num_epochs}"
            print(epoch_formatted)
            train_log += "\n" + epoch_formatted

            time_start = time.time()
            self.train_epoch(dataloader_train, optim)

            train_perf = self.perf(dataloader_train)
            train_log += "\n" + self.perf_report("Train", train_perf)
            train_perf_all.append(train_perf)

            val_perf = self.perf(dataloader_val)
            train_log += "\n" + self.perf_report("Valid", val_perf)
            val_perf_all.append(val_perf)

            val_loss = val_perf[0]
            lr_scheduler.step(val_loss)

            if best_loss - val_loss >= epoch_thresh:
                epoch_patience_counter = 0
                best_loss = val_loss
                if save_best:
                    best_weights = self.state_dict()
            else:
                epoch_patience_counter += 1

            time_end = time.time()
            duration = time_end - time_start
            duration_formatted = f"Time: {duration:.2f}s"
            print(duration_formatted)
            train_log += "\n" + duration_formatted

            if tune_thresh_every is not None and epoch % tune_thresh_every == 0:
                train_log += "\n" + f"\nTuning location probability threshold ..."
                print(f"\nTuning location probability threshold ...")
                thresh = self.get_loc_prob_thresh()
                self.tune_loc_prob_thresh(dataloader_train, verbose=False)
                train_log += f"Threshold: {thresh:.1f}% --> {self.get_loc_prob_thresh():.1f}%"
                print(f"Threshold: {thresh:.1f}% --> {self.get_loc_prob_thresh():.1f}%")

            if epoch_patience is not None and epoch_patience_counter == epoch_patience:
                ending = f"\nEnding training early because validation loss has not increased in {epoch_patience} epochs"
                train_log += "\n" + ending
                print(ending)
                break

        self.logs["train.log"] = train_log
        self.logs["train_perf.npy"] = np.vstack(train_perf_all)
        self.logs["val_perf.npy"] = np.vstack(val_perf_all)

        if save_best:
            print("Loading best weights ...")
            self.load_state_dict(best_weights)

        train_log += "\n" + f"\nTuning location probability threshold ..."
        print(f"\nTuning location probability threshold ...")
        thresh = self.get_loc_prob_thresh()
        self.tune_loc_prob_thresh(dataloader_train, verbose=False)
        train_log += f"Threshold: {thresh:.1f}% --> {self.get_loc_prob_thresh():.1f}%"
        print(f"Threshold: {thresh:.1f}% --> {self.get_loc_prob_thresh():.1f}%")

        train_end = time.time()

        perf = self.perf(dataloader_val, plot_preds=())
        perf_report = self.perf_report("\nFinal Val", perf)
        train_log += "\n" + perf_report

        train_log += "\n\n" + f"Time: {train_end-train_start:.1f}s"

        val_f1_score_final = perf[5]
        return train_perf_all, val_perf_all, val_f1_score_final

    def set_loc_prob_thresh(self, loc_prob_thresh):
        """
        loc_prob_thresh is in (0, 100)
        internally, self.loc_prob_thresh_logit is (-inf, inf) since model's outputs are not from sigmoid
        """
        self.loc_prob_thresh_logit = torch.logit(torch.tensor(loc_prob_thresh/100)).item()

    def get_loc_prob_thresh(self):
        """
        loc_prob_thresh is in (0, 100)
        internally, self.loc_prob_thresh_logit is (-inf, inf) since model's outputs are not from sigmoid
        """
        return torch.sigmoid(torch.tensor(self.loc_prob_thresh_logit)).item() * 100

    def loc_to_logit(self, loc):
        # Normalize index of waveform to model's localization logit
        return (loc - self.loc_first_frame)  # .to(torch.long)

    def logit_to_loc(self, logit):
        # Denormalize model's localization logit to index of waveform
        if isinstance(logit, torch.Tensor):
            logit = logit.cpu().numpy()

        return logit + self.loc_first_frame

    def outputs_to_preds(self, outputs, return_wf_count=False):
        """
        Convert raw model outputs to predictions

        :param outputs: torch.Tensor
            Direct outputs of forward call of model
        :param return_wf_count: bool
        :return:
            If return_wf_count == True, returns tuple of (preds, number of waveforms predicted)
            If return_wf_count == False, returns only preds

            preds is a list where len(preds) == len(outputs). Each element of preds is a np.array
            where each element in this np.array is the location of a predicted waveform
        """

        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()

        if return_wf_count:
            preds = []
            wf_count = 0
            for i in range(len(outputs)):
                peaks = self.logit_to_loc(find_peaks(np.concatenate(((-np.inf,), (outputs[i]), (-np.inf,))), height=self.loc_prob_thresh_logit)[0])
                peaks -= 1
                preds.append(peaks)
                wf_count += len(peaks)
            return preds, wf_count
        else:
            preds = [
                self.logit_to_loc(
                    find_peaks(
                        np.concatenate(((-np.inf,), (outputs[i]), (-np.inf,))),  # Pad beginning and end with -inf so that first location frame and last location frame can be identifed as peaks
                        height=self.loc_prob_thresh_logit
                    )[0] - 1  # Subtract one to account for the np.concatenate
                )
                for i in range(len(outputs))
            ]
            return preds
        # preds when only one waveform can be in a single sample
        # preds[:, 0] = np.round(outputs[:, self.idx_spike])  # spike detection
        # preds[:, 0] = outputs[:, self.idx_spike] > 0.3

        # highest_probs = torch.softmax(torch.as_tensor(outputs[:, self.idx_loc]), dim=1).max(dim=1)[0]
        # preds[:, 0] = highest_probs.cpu().numpy() >= self.loc_prob_thresh
        #
        # preds[:, 1] = self.logit_to_loc(outputs[:, self.idx_loc])  # spike localization
        #
        # preds[:, 2] = self.logit_to_alpha(outputs[:, self.idx_alpha])  # spike clustering (distinguishing between spikes)

    def perf(self, dataloader, loc_buffer=8,
             plot_preds=(), max_plots=10,
             outputs_list=None):
        """
        Get performance stats with data based on data in dataloader

        :param dataloader
        :param loc_buffer: int
            For accuracy, recall, and precision:
                st = correct a waveform's correct location
                st_region = interval of [st-loc_buffer, st+loc_buffer]
                    A node being in st_region is the same as the location_of_node - st <= loc_buffer
                A location node is one of the model's output nodes that corresponds to a possible location of a waveform.
                A location node predicts a waveform if its probability of a waveform > self.loc_prob_thresh
                Each predicted waveform only counts for one label waveform and vice versa
                    (i.e. if there is a true positive, the rest of the waveforms are evaluated as if the waveforms in the true positive do not exist)

                True positive = location node in st_region predicts a waveform
                False positive = location node not in st_region predicts a waveform
                True negative = location node not in st_region does not predict a waveform
                False negative = location node in st_region does not predict a waveform

        :param plot_preds:
            If "correct" in plot_preds, samples that were correctly classified will be plotted
            If "failed" in plot_preds, samples that were incorrectly classified will be plotted
            If "all" in plot_preds, all samples will be plotted
            If "hist" in plot_preds, location MAD percent error histograms will be plotted
        :param max_plots:
            Maximum number of plots if plotting
            If None, no max

        :param outputs_list: list or None
            If None, outputs will be calculated
            If a list, contains the outputs of iterating through :param dataloader: in order

        :returns: tuple
            1) loss
            2) % wf detected (num wf predicted by model / num actual wf * 100)
            3) % accuracy
            4) % recall
            5) % precision
            6) Loc MAD between location of waveforms predicted by model and label waveforms (in frames)
            7) Loc MAD (in ms)
        """
        plot_preds = {plot_preds} if isinstance(plot_preds, str) else set(plot_preds)
        num_plots = 0

        self.train(False)
        with torch.no_grad():
            num_samples = 0
            loss_total = 0
            num_wf_pred_all = 0  # Total number of waveforms predicted by model
            num_wf_pred_correct = 0  # Number of correctly predicted waveforms by model
            num_wf_label = 0  # Total number of actual waveforms
            num_frames_total = 0  # Total number of time frames with a potential waveform that model predicts for

            loc_deviations = []

            above_dists = []  # Distances of false positives above probability threshold
            below_dists = []  # Distances of false negatives below probability threshold

            # utils.random_seed(231, silent=True)

            for i, (inputs, num_wfs, wf_locs, wf_alphas) in enumerate(dataloader):
                if outputs_list is None:
                    outputs = self(inputs)
                else:
                    outputs = outputs_list[i]
                
                # if num_wfs > 0:
                loss_total += self.loss(outputs, num_wfs, wf_locs).item()
                num_samples += 1
                num_frames_total += torch.numel(outputs)

                # Performance when multiple waveforms can exist in a sample
                preds = self.outputs_to_preds(outputs, return_wf_count=False)

                for j, (loc_preds, num_wf, loc_labels) in enumerate(zip(preds, num_wfs, wf_locs.cpu().numpy())):
                    wf_count = len(loc_preds)
                    num_wf_pred_all += wf_count

                    num_wf = num_wf.item()  # num_wf is the correct number of waveforms
                    num_wf_label += num_wf

                    if ("all" in plot_preds) \
                            or ("correct" in plot_preds and wf_count == num_wf) \
                            or ("failed" in plot_preds and wf_count != num_wf) \
                            or ("noise" in plot_preds and wf_count == 0):
                        if max_plots is None or (max_plots is not None and num_plots < max_plots):
                            self.plot_pred(inputs[j, 0, :], outputs[j], loc_preds,
                                           num_wf, loc_labels, wf_alphas[j],
                                           dataloader)
                            num_plots += 1                                
                    
                    wf_true_positives = set()  # Store which pred waveforms have already been assigned to a label waveforms
                    labels_predicted = set()  # Store which label waveforms have already been assigned to a pred waveform
                                
                    pairs_dists = []  # each element is distance between a loc_pred and loc_label
                    pairs_ind = []  # each element is (loc_pred_idx, loc_label_ind)    
                    for idx_pred in range(len(loc_preds)):
                        for idx_label in range(num_wf):
                            pairs_dists.append(np.abs(loc_preds[idx_pred] - loc_labels[idx_label]))
                            pairs_ind.append((idx_pred, idx_label))
                            
                    order = np.argsort(pairs_dists)  # Mark as TP the predicted waveforms closest to label waveform
                    for o in order:
                        dist = pairs_dists[o]
                        idx_pred, idx_label = pairs_ind[o]
                        if idx_pred in wf_true_positives:
                            continue
                        if idx_label in labels_predicted:
                            continue
                        
                        if dist <= loc_buffer: # label was detected:
                            loc_deviations.append(dist)
                            wf_true_positives.add(idx_pred)
                            labels_predicted.add(idx_label)
                        else:  # pairs_dists is sorted ascending, so if dist is above threshold, all following are above too
                            break
                                
                    num_wf_pred_correct += len(wf_true_positives)

                    # Find distances of false positive probability predictions above prediction threshold
                    for i_pred in range(len(loc_preds)):
                        if i_pred not in wf_true_positives:  # False positive
                            logit_frame = self.loc_to_logit(loc_preds[i_pred])
                            logit = outputs[j, logit_frame].item()
                            dist = sigmoid(logit)*100 - self.get_loc_prob_thresh()
                            above_dists.append(dist)
                            
                    # Find distances of false negative probability predictions below prediction threshold
                    for i_label in range(num_wf):
                        if i_label not in labels_predicted:  # False negative
                            logit_frame = self.loc_to_logit(loc_labels[i_label])
                            logit = outputs[j, logit_frame].item()
                            dist = self.get_loc_prob_thresh() - sigmoid(logit)*100
                            below_dists.append(dist)

            loc_mad_frames = np.mean(loc_deviations) if len(loc_deviations) > 0 else np.nan
            loc_mad_ms = utils.frames_to_ms(loc_mad_frames)

            if "hist" in plot_preds:
                # Plot histogram of absolute deviation of locations
                plot.plot_hist_loc_mad(utils.frames_to_ms(np.array(loc_deviations)))

                # Plot histogram of percent absolute error
                # plot.plot_hist_percent_abs_error(alpha_percent_abs_errors)

            recall = 100 * num_wf_pred_correct / num_wf_label if num_wf_label > 0 else np.nan
            precision = 100 * num_wf_pred_correct / num_wf_pred_all if num_wf_pred_all > 0 else np.nan
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else np.nan

            # plt.hist(above_dists, bins=10)
            # print(np.median(above_dists))
            # print(np.mean(above_dists))
            # plt.ylabel("Number of false positives")
            # plt.xlabel("Prediction percent above spike detection threshold")
            # plt.xlim(0)
            # plt.show()
            
            # plt.hist(below_dists, bins=10)
            # print(np.median(below_dists))
            # print(np.mean(below_dists))
            # plt.ylabel("Number of false negatives")
            # plt.xlabel("Prediction percent below spike detection threshold")
            # plt.xlim(0)
            # plt.show()
            
            return (
                loss_total / num_samples,  # Loss
                100 * num_wf_pred_all / num_wf_label if num_wf_label > 0 else np.nan,  # Ratio of number of waveforms predicted by model to number of correct waveforms
                100 * (num_wf_pred_correct + num_frames_total - (num_wf_pred_all + num_wf_label - num_wf_pred_correct)) / num_frames_total,  # Accuracy
                recall,
                precision,
                f1_score,
                loc_mad_frames,
                loc_mad_ms
            )
            # return stats

    def plot_pred(self, trace:torch.Tensor, output, pred,
                  num_wf, wf_labels, wf_alphas,
                  multi_rec=None):

        """
        Plot models prediction for a sample
        
        :param multi_rec: The MultiRecordingDataset (or dataloader) that generated sample
            If None: Don't plot underlying waveform in trace
            Else: Plot underlying waveform in trace
        """
        ALPHA = 0.7
        LINESTYLE = "dashed"

        if isinstance(trace, torch.Tensor):
            trace = trace.cpu().numpy().flatten()
        if isinstance(wf_labels, torch.Tensor):
            wf_labels = wf_labels.cpu().numpy()
        if isinstance(wf_alphas, torch.Tensor):
            wf_alphas = wf_alphas.cpu().numpy()

        fig, (a0, a1, a2) = plt.subplots(3, tight_layout=True, figsize=(7, 7))
        subplots = (a0, a1, a2)

        # Set yticks, ylim, xlim, xlabel
        plot.set_ticks(subplots, trace)

        # Plot trace
        a1.set_title("Model Input")
        # rms = np.sqrt(np.mean(np.square(trace)))
        # filtered = bandpass_filter(trace)
        # rms = np.sqrt(np.mean(np.square(filtered)))
        # rms = 3.13
        # a1.axhline(5 * rms, linestyle="dashed", color="black", linewidth=1, alpha=0.5)  # , label="5 RMS"
        # a1.axhline(-5 * rms, linestyle="dashed", color="black", linewidth=1, alpha=0.5)
        a1.plot(trace)  # , label=f"{rms:.1f}")

        false_positives = set(pred)  # Initially, false positives is every prediction
        plot_false_negative = False

        # Plot correct wf
        if num_wf:
            # Plot each waveform
            for loc, alpha in zip(wf_labels, wf_alphas):
                if loc == -1 or alpha == np.inf:  # There is no wf
                    continue

                if loc in false_positives:  # If label matches with a prediction, the prediction is a TP not a FP
                    false_positives.remove(loc)
                    color = "green"
                else:
                    color = "blue"
                    plot_false_negative = True

                # Plot location of waveform in trace
                a1.axvline(loc, alpha=ALPHA, color=color, linestyle=LINESTYLE)

                # Plot underlying waveform on separate axis
                if multi_rec is not None:
                    loc = int(loc)  # Needs to be int for plotting waveform and removing waveform from trace

                    # a0.axvline(loc, alpha=ALPHA, color=color, linestyle=LINESTYLE)
                    wf, peak_idx, wf_len, _, _ = multi_rec.wfs[alpha].unravel()
                    plot.plot_waveform(wf, peak_idx, wf_len, alpha, loc, a0)

        # Plot location of predicted waveforms
        for i, loc in enumerate(false_positives):
            a1.axvline(loc, alpha=ALPHA, color="red", linestyle=LINESTYLE, label="FP" if i == 0 else None)

        # Create legend for labels for vertical lines if they are in plot
        if len(false_positives) < len(pred):
            a1.axvline(-1000, alpha=ALPHA, color="green", linestyle=LINESTYLE, label="TP")
        if plot_false_negative:
            a1.axvline(-1000, alpha=ALPHA, color="blue", linestyle=LINESTYLE, label="FN")

        if num_wf > 0 and multi_rec is not None:
            a0.legend()
        if len(pred) > 0 or num_wf > 0:
            a1.legend()

        # Plot location probabilities
        self.plot_loc_probs(output, a2)

        plt.show()

    def plot_loc_probs(self, model_output, axis):
        # Plot distribution of model's location probabilities of single output
        # :param model_output: is the raw outputs of the model
        # axis is a plt subplot

        output = torch.sigmoid(model_output.to(torch.float32))
        axis.plot(np.arange(len(output)) + self.loc_first_frame, output.cpu() * 100, color="red")
        axis.axhline(self.get_loc_prob_thresh(), linestyle="dashed", color="black", label="Detection Threshold", linewidth=1)
        axis.set_title("Location probabilities")
        axis.set_ylim(0, 100)
        axis.set_yticks(range(0, 101, 20), [f"{p}%" for p in range(0, 101, 20)])
        axis.legend()

    def save(self, folder, logs=(), src=r"/data/MEAprojects/DLSpikeSorter/src"):
        """
        In folder, saves another folder with time it was created
        which contains all relevant info about the model in the following hierarchy:
            folder
                yymmdd_HHMMSS_ffffff (see utils.get_time for more details)
                    state_dict.pt: model's PyTorch parameters (weights, biases, etc)
                    init_dict.json: model's init args
                    src: All source code in src folder (except __init__.py) that are needed to recreate and run model
                        data.py
                        model.py
                        plot.py
                        train.py
                        utils.py
                    log: .log, .txt, and any other data files specified in :param logs:

        :param folder: Path or str
            Folder to save model's folder in
        :param logs: tuple
            Each element is a tuple of (log_file_name, log_file_contents) to save
        :param src: Path or str
            Path to folder containing model's source code

        :return: str
            Name of model (time it was created)

        """
        name = utils.get_time()
        folder_model = Path(folder) / name # utils.get_time()
        folder_src = folder_model / "src"
        folder_log = folder_model / "log"
        folder_src.mkdir(parents=True, exist_ok=True)
        folder_log.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), folder_model / "state_dict.pt")

        init_dict = {
            "num_channels_in": self.num_channels_in,
            "sample_size": self.sample_size, "buffer_front_sample": self.buffer_front_sample, "buffer_end_sample": self.buffer_end_sample,
            "loc_prob_thresh": self.get_loc_prob_thresh(), "buffer_front_loc": self.buffer_front_loc, "buffer_end_loc": self.buffer_end_loc,
            "input_scale": self.input_scale,
            "device": str(self.device),
            "architecture_params": self.architecture_params,
        }
        with open(folder_model / "init_dict.json", 'w') as f:
            json.dump(init_dict, f)

        for f in Path(src).iterdir():
            if f.suffix == ".py" and f.name != "__init__.py":
                if f.name == "train.py":
                    print("Not copying train.py")
                else:
                    utils.copy_file(f, folder_src)
                    
        for file, contents in logs:
            if file.endswith("npy"):
                np.save(folder_log / file, contents)
            else:
                with open(folder_log / file, "w") as f:
                    f.write(contents)

        for file, contents in self.logs.items():
            if file.endswith("npy"):
                np.save(folder_log / file, contents)
            else:
                with open(folder_log / file, "w") as f:
                    f.write(contents)

        return name

    def tune_loc_prob_thresh(self, dataloader, start=None, stop=50, step=2.5,
                             verbose=True, outputs_list=None):
        """
        Set self.loc_prob_thresh to value that gives best weighted sum of recall and precision

        :param dataloader:
            Data to set value for
        :param start: percent
            If None, start = step
            Test the thresholds in [start, last] with step
        :param step: percent
            Test the thresholds in [step, last] with step
        :param stop: percent
            Test the thresholds in [start, last] with step
        :param verbose:
            If True, print results
        :param outputs_list: list or None
            If None, outputs will be calculated
            If a list, contains the outputs of iterating through :param dataloader: in order
        """
        start = step if start is None else start

        # Get outputs of model
        self.train(False)

        if outputs_list is None:
            inputs_list = []
            outputs_list = []
            with torch.no_grad():
                for inputs, num_wfs, wf_locs, wf_alphas in dataloader:
                    inputs_list.append((inputs, num_wfs, wf_locs, wf_alphas))
                    outputs_list.append(self(inputs))
        else:
            inputs_list = dataloader

        # Find best score
        best_score = 0
        best_thresh = self.loc_prob_thresh_logit
        num = int(stop // step)
        for thresh in np.linspace(start, num * step, num):
            self.set_loc_prob_thresh(thresh)

            perf = self.perf(inputs_list, outputs_list=outputs_list)

            f1_score = perf[5]
            if verbose:
                self.perf_report(f"Prob Thresh: {thresh:.1f}%", perf)
                print(f"F1 Score: {f1_score:.1f}\n")
            if f1_score > best_score:
                best_score = f1_score
                best_thresh = thresh
        if verbose:
            print(f"Best thresh: {best_thresh:.1f}%")
        self.set_loc_prob_thresh(best_thresh)

    def log(self, path, save_data):
        """
        Save save_data to model_path/log/path

        :param path:
        :param save_data:
        :return:
        """

        if self.path is None:
            raise ValueError("model's path is not set. Set it with 'model.path = PATH'")

        path = Path(self.path) / "log" / path
        path.parent.mkdir(parents=True, exist_ok=True)

        if Path(path).suffix == ".npy":
            np.save(path, save_data)
        else:
            raise NotImplementedError("data format not implemented yet")

    def compile(self, n_dim_0: int, model_save_path=None,
                dtype=torch.float16, device="cuda"):
        """
        Compile model with torch tensorrt 
        
        Params:
        n_dim_0
            Input to model should be (n_dim_1, self.num_channels_in, self.sample_size)
        
        model_save_path
            If not None, save compiled model to model_save_path/compiled.ts (useful to cache since compiling can take a long time)
        """
        
        # can use random example data: https://apple.github.io/coremltools/docs-guides/source/model-tracing.html#:~:text=For%20an%20example%20input%2C%20you,input%2C%20needed%20by%20jit%20tracer.
        # some versions of the code used to create figures did not set the random seed for this compiling, so results may be different 
        np.random.seed(231)
        torch.manual_seed(231)
        
        model = self.model.conv
        model.to(device=device, dtype=dtype)
        model = torch.jit.trace(model, [torch.rand(n_dim_0, self.num_channels_in, self.sample_size, dtype=dtype, device=device)])
        model = torch_tensorrt.compile(model, 
                                       inputs=[torch_tensorrt.Input((n_dim_0, self.num_channels_in, self.sample_size), dtype=dtype)], 
                                       enabled_precisions={dtype},
                                       ir="ts")

        if model_save_path is not None:
            torch.jit.save(model, model_save_path / ModelSpikeSorter.compiled_name)
            
    @staticmethod
    def load_compiled(model_save_path):
        """
        Load saved compiled model
        """
        return torch.jit.load(model_save_path / ModelSpikeSorter.compiled_name)
    
    @staticmethod
    def load(folder):
        
        # Load model from :param folder:

        folder = Path(folder)
        if not (folder / "init_dict.json").exists():
            """
            Model data hierarchy
                version (v0_2)
                    holdout recording (2953)
                        yymmdd_HHMMSS (220928_012416) = model's folder

            If folder == holdout recording, return most recent model (latest date)
            """
            folder = sorted(folder.iterdir())[-1]

        with open(folder / "init_dict.json", 'r') as f:
            init_dict = json.load(f)
        model = ModelSpikeSorter(**init_dict)
        model.load_state_dict(torch.load(folder / 'state_dict.pt'))
        model.path = folder
        return model

    @staticmethod
    def get_output_shape(layer, input_shape, device="cpu", dtype=torch.float32):
        layer = layer.to(device=device, dtype=dtype)
        return layer(torch.zeros(input_shape, device=device, dtype=dtype)).shape

    @staticmethod
    def get_same_padding(conv_kwargs):
        """Get padding layer analogous to TensorFlow's SAME padding (output size is same as input IFF stride=1)"""
        total = conv_kwargs["kernel_size"] - 1
        left = total // 2
        right = total - left
        return nn.ConstantPad1d((left, right), 0.0)

    @staticmethod
    def perf_report(preface, perf):
        report = ModelSpikeSorter._perf_report.format(preface, *perf)
        print(report)
        return report


class ModelTuning(nn.Module):
    def __init__(self, architecture, num_channels,
                 relu, add_conv, bottleneck, noise, filter,
                 sample_size=200):
        super().__init__()

        num_layers, kernel_size = self.parse_architecture(architecture)

        get_relu = lambda num_parameters: nn.ReLU() if relu == "relu" else nn.PReLU(num_parameters)

        if num_layers is not None:
            conv = nn.Sequential()
            in_channels = 1
            out_channels = num_channels
            skip_relu = False
            for i in range(num_layers):
                if i == num_layers-1 and not add_conv and bottleneck == 0:  # If last layer
                    out_channels = 1
                    if noise == 0:
                        skip_relu = True

                conv.append(nn.Conv1d(in_channels, out_channels, kernel_size))
                if not skip_relu:
                    conv.append(get_relu(out_channels))
                in_channels = out_channels

            if add_conv > 0:
                for i in range(add_conv):
                    if i == add_conv - 1 and bottleneck == 0:  # If last layer
                        out_channels = 1
                        if noise == 0:
                            skip_relu = True

                    conv.append(nn.Conv1d(in_channels, out_channels, 3, padding=1))
                    if not skip_relu:
                        conv.append(get_relu(out_channels))
                    in_channels = out_channels

            if bottleneck == 1:
                conv.append(nn.Conv1d(in_channels, 1, 1))
                if noise != 0:
                    conv.append(get_relu(1))
            self.last_layer = list(conv.modules())[-1]
        else:
            conv = UNet()
            self.last_layer = conv.last
        self.conv = conv

        if noise == 0 or isinstance(conv, UNet):
            self.noise = nn.Flatten()
        else:
            noise_conv = nn.Conv1d(2, 1, 1)
            self.last_layer = noise_conv
            if noise == 1:
                self.noise = nn.Sequential(noise_conv, nn.Flatten())
            elif noise == 0.5:  # Linear layers to model noise
                noise_linear = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(sample_size, sample_size),
                    get_relu(sample_size),
                    nn.Linear(sample_size, 1),
                )
                noise_sequential = nn.Sequential(noise_conv, nn.Flatten())
                self.noise = nn.ModuleList([noise_linear, noise_sequential])
            elif noise == 0.75:   # Conv layers to model noise:
                pass

        self.filter = data.BandpassFilter((300, 3000)) if filter else None

    def forward(self, x):
        x2 = self.conv(x)
        if isinstance(self.noise, nn.Flatten):
            x2 = self.noise(x2)
        elif isinstance(self.noise, nn.Sequential):
            rms = torch.sqrt(torch.mean(torch.square(x), dim=(1, 2), keepdim=True)) - 1.3  # 1.3 is mean
            x2 = torch.cat([x2, rms.repeat(1, 1, x2.shape[-1])], dim=1)
            x2 = self.noise(x2)
        elif isinstance(self.noise, nn.ModuleList):
            x_noise = self.noise[0](x)[:, :, None].repeat(1, 1, x2.shape[-1])
            x2 = torch.cat([x2, x_noise], dim=1)
            x2 = self.noise[1](x2)
        return x2

    def init_final_bias(self, num_output_locs: int, num_wfs_probs: list):
        """
        Initialize bias of the final layer based on the waveform probabilities of training dataset
        (assumes 50% of samples contain no waveform)

        :param num_wfs_probs:
            If there is at least 1 waveform in the sample, then the probability of i additional waveforms occurring
            in the sample is num_wfs_probs[i]
        """
        # Get the last weight layer
        # last_weight_layer = [module for module in self.modules() if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear)][-1]
        last_weight_layer = self.last_layer

        # Get the probability of a waveform occurring at a location probability
        exp_prob = 0
        for i, prob in enumerate(num_wfs_probs):
            exp_prob += prob * (i + 1)
        exp_prob *= 0.5 * 1/num_output_locs  # 50% chance of a waveform appearing at all, and 1/num_output_locs for waveform appearing at a output location
        # torch.sigmoid(bias) = exp_prob
        nn.init.constant_(last_weight_layer.bias, torch.logit(torch.tensor(exp_prob)).item())

    @staticmethod
    def parse_architecture(architecture):
        """
        Convert architecture number to num_layers and kernel_size
        
        negative p:architecture: is for neuropixels (30kHz). positive is for MEA (20kHz)
        """
        # Convert architecture number to num_layers and kernel_size

        # Output receptive field = 60
        if architecture == 1:
            # 4 conv layers of 16/1
            NUM_LAYERS = 4
            KERNEL_SIZE = 16
        elif architecture == 2:
            # 6 conv layers of 11/1
            NUM_LAYERS = 6
            KERNEL_SIZE = 11
        elif architecture == 3:
            # 10 filters of 7/1
            NUM_LAYERS = 10
            KERNEL_SIZE = 7

        # Output receptive field = 80
        elif architecture == 4:
            # 4 conv layers of 21/1
            NUM_LAYERS = 4
            KERNEL_SIZE = 21
        elif architecture == -4:
            NUM_LAYERS = 4
            KERNEL_SIZE = 31
        elif architecture == 5:
            # 8 conv layers of 11/1
            NUM_LAYERS = 8
            KERNEL_SIZE = 11
        elif architecture == 6:
            # 10 conv layers of 9/1
            NUM_LAYERS = 10
            KERNEL_SIZE = 9

        # Misc.
        elif architecture == 7:
            # UNet
            NUM_LAYERS = None
            KERNEL_SIZE = None
        else:
            raise ValueError("Invalid architecture parameter")

        return NUM_LAYERS, KERNEL_SIZE


class UNet(nn.Module):
    # sample_size: 204, front_buffer: 44, end_buffer: 44
    def __init__(self, depth=4, first_conv_channels=32):
        super().__init__()

        self.contracting = nn.ModuleList()
        in_channels = 1
        out_channels = first_conv_channels
        for i in range(depth):
            self.contracting.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels
            out_channels *= 2
        self.pool = nn.MaxPool1d(2, 2)

        self.expanding = nn.ModuleList()
        in_channels_x = out_channels // 2
        for i in range(depth - 1):
            self.expanding.append(ExpandBlock(in_channels_x))
            in_channels_x //= 2

        self.last = nn.Conv1d(in_channels_x, 1, 1)

    def forward(self, x):
        copies = []
        for layer in self.contracting[:-1]:
            x = layer(x)
            copies.append(x)
            x = self.pool(x)
            # print(x.shape)
        x = self.contracting[-1](x)
        # print(x.shape)

        # print()

        for copy, layer in zip(copies[::-1], self.expanding):
            x = layer(x, copy)
            # print(x.shape)

        return self.last(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class ExpandBlock(nn.Module):
    def __init__(self, in_channels_x, kernel_size=3):
        super().__init__()
        self.up_conv = nn.ConvTranspose1d(in_channels_x, in_channels_x // 2, 2, 2)
        self.relu = nn.ReLU()
        self.conv_block = ConvBlock(in_channels_x, in_channels_x // 2, kernel_size)

    def forward(self, x, cat):
        up = self.relu(self.up_conv(x))
        x = torch.cat([self.crop(cat, up.shape[2]), up], dim=1)
        return self.conv_block(x)

    @staticmethod
    def crop(x, size_out):
        size = x.shape[2]
        cropped = size - size_out
        left = int(cropped // 2)
        return x[:, :, left:-left - cropped % 2]


class RMSThresh(nn.Module):
    """
    Model where spikes are classified based on RMS threshold
    """
    def __init__(self, thresh=5, sample_size=200, buffer_front=40, buffer_end=40):
        super().__init__()

        self.thresh = thresh
        self.filter = data.BandpassFilter((300, 3000))
        #self.sample_size = sample_size
        self.buffer_front = buffer_front
        self.buffer_end = buffer_end

    def forward(self, x):
        x /= 100
        dtype = x.dtype
        device = x.device
        x = x.cpu()
        x = torch.tensor(self.filter(x[:, 0, :]).copy())
        rms = torch.sqrt(torch.mean(torch.square(x), keepdim=True, dim=-1))

        return (torch.abs(x[:, self.buffer_front:-self.buffer_end]) >= self.thresh * rms).to(dtype=dtype, device=device) * 200 - 100


def sigmoid(x):
    # return np.where(x>=0,
    #                 1 / (1 + np.exp(x)),
    #                 np.exp(x) / (1+np.exp(x))
    #                 )
    return np.exp(x) / (1+np.exp(x))  # Positive overflow is not an issue because DL does not output large positive values (only large negative)


def main():
    BATCH_SIZE = 2000
    NUM_CHANNELS_IN = 1
    SAMPLE_SIZE = 80

    inputs = torch.rand(BATCH_SIZE, NUM_CHANNELS_IN, SAMPLE_SIZE, device="cpu", dtype=torch.float32)

    model = ModelSpikeSorter(1, SAMPLE_SIZE, 0, 0, 35, 0, 0, "cpu", (4, 1, "relu", 0, 0, 0, 0))

    # model = ModelTuning(*ModelTuning.parse_architecture(4),
    #                     relu="prelu", add_conv=4, bottleneck=1, noise=0, filter=0,
    #                     sample_size=SAMPLE_SIZE).to("cuda")
    # from torchsummary import summary
    # summary(model, (NUM_CHANNELS_IN, SAMPLE_SIZE))

    print(*inputs.shape)
    print(*model(inputs).shape)


if __name__ == "__main__":
    main()