# DISCLAIMER:
# The integration of different HW models is currently not impemented,
# the proposed MPIC model is only an example but the current implementation
# directly support only `diana`

# TODO: Understand how model changes for deptwhise conv.
#       At this time groups is not taken into account!

import math
import torch
from flexnas.methods.pit.nn.binarizer import PITBinarizer
from torchinfo import summary

def _floor(ch, N):
    return math.floor((ch + N - 1) / N)


def compute_layer_latency_GAP8(self):
    """Computes the MACs of each possible layer of the PITSuperNetModule
    and stores the values in a list.
    It removes the MACs of the PIT modules contained in each layer because
    these MACs will be computed and re-added at training time.
    """
    for layer in self.sn_input_layers:
        stats = summary(layer, self.input_shape, verbose=0, mode='eval')
        import pdb;pdb.set_trace()
        self.layers_macs.append(stats.total_mult_adds)


def get_latency_conv2D_GAP8_supernet(self) -> torch.Tensor:
    """Method that computes the number of MAC operations for the module

    :return: the number of MACs
    :rtype: torch.Tensor
    """
    soft_alpha = torch.argmax(self.alpha, dim=0)

    latency = torch.tensor(0, dtype=torch.float32)
    for i in range(self.n_layers):
        latency = latency + (soft_alpha[i] * (self.layers_latency[i]))
    return latency

def get_latency_conv2D_GAP8(self) -> torch.Tensor:
    iterations = _floor(self.out_height, 2) * _floor(self.out_width, 8)
    cin_mask = self.input_features_calculator.features_mask
    ch_in = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
    cout_mask = self.out_features_masker.theta
    ch_out = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
    im2col = self.kernel_size[0] * self.kernel_size[1] * ch_in * 2
    matmul = _floor(ch_out, 4) * (5 + _floor(self.kernel_size[0] * self.kernel_size[1] * ch_in, 4) * (6 + 8) + 10)
    latency = iterations * (im2col + matmul)
    return latency

def get_latency_Linear_GAP8(self) -> torch.Tensor:
    cin_mask = self.input_features_calculator.features_mask
    ch_in = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
    # Compute actual integer number of output channels
    cout_mask = self.out_features_masker.theta
    ch_out = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
    # Finally compute cost
    latency = ch_in * ch_out
    return latency

def get_latency_conv2D_Diana(self) -> torch.Tensor:
    cin_mask = self.input_features_calculator.features_mask
    ch_in = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
    cout_mask = self.out_features_masker.theta
    ch_out = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
    cycles = _floor(ch_out / self.groups, 16) * ch_in * _floor(self.out_height, 16) * self.out_width * self.kernel_size[0] * self.kernel_size[1]
    # Works with both depthwise and normal conv:
    cycles_load_store = self.out_height * self.out_width * (ch_out + ch_in) / 8
    return cycles_load_store + cycles


def get_latency_Linear_Diana(self) -> torch.Tensor:
    cin_mask = self.input_features_calculator.features_mask
    ch_in = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
    # Compute actual integer number of output channels
    cout_mask = self.out_features_masker.theta
    ch_out = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
    # Finally compute cost
    cycles = _floor(ch_out, 16) * ch_in
    # Works with both depthwise and normal conv:
    cycles_load_store = (ch_out + ch_in) / 8
    latency = ch_in * ch_out
    return latency

def get_latency_total(self) -> torch.Tensor:
    latency = torch.tensor(0, dtype=torch.float32)
    # size = torch.tensor(0)
    for layer in self._target_layers:
        # size += layer.get_size()
        latency = latency + layer.get_latency()
    return latency