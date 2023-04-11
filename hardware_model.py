# DISCLAIMER:
# The integration of different HW models is currently not impemented,
# the proposed MPIC model is only an example but the current implementation
# directly support only `diana`

# TODO: Understand how model changes for deptwhise conv.
#       At this time groups is not taken into account!

from typing import List, cast, Iterator, Tuple, Any, Dict
import math
import torch
import torch.nn as nn
from flexnas.methods.pit.nn.binarizer import PITBinarizer
from torchinfo import summary
from flexnas.methods.pit.nn import PITModule

class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ch, N):
        return torch.floor((ch + N - 1) / N)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    
def _floor(ch, N):
    return math.floor((ch + N - 1) / N)

def compute_layer_latency_GAP8(self, input_shape):
    """Computes the MACs of each possible layer of the PITSuperNetModule
    and stores the values in a list.
    It removes the MACs of the PIT modules contained in each layer because
    these MACs will be computed and re-added at training time.
    """
    for layer in self.sn_input_layers:
        stats = summary(layer, input_shape, verbose=0, mode='eval')
        latency = 0
        try:
            for obj in layer:
                if obj.__class__.__name__ == "ConvBlock_MN":
                    obj = obj.conv1
                if obj.__class__.__name__ == "Conv2d":
                    ch_in = obj.weight.shape[1]
                    ch_out = obj.weight.shape[0]
                    kernel_size_x = obj.weight.shape[2]
                    kernel_size_y = obj.weight.shape[3]
                    iterations = _floor(int(input_shape[2]/obj.stride[0]), 2) * _floor(int(input_shape[3]/obj.stride[0]), 8)
                    im2col = kernel_size_x * kernel_size_y * ch_in * 2
                    matmul = _floor(ch_out, 4) * (5 + _floor(kernel_size_x * kernel_size_y * ch_in, 4) * (6 + 8) + 10)
                    latency += iterations * (im2col + matmul)
                    if obj.groups > 1:
                        # 1 MAC/cycle
                        latency = 4 * _floor(ch_out, 4) * int(input_shape[2]/obj.stride[0]) * int(input_shape[3]/obj.stride[0]) * kernel_size_x * kernel_size_y
                elif obj.__class__.__name__ == "Identity":
                    latency += 0
                elif obj.__class__.__name__ == "Linear":
                    ch_in = obj.weight.shape[1]
                    ch_out = obj.weight.shape[0]
                    latency += _floor(ch_in, 2) * _floor(ch_out, 4)
        except:
            if layer.__class__.__name__ == "Identity":
                obj = layer
            else:
                obj = layer.conv1
            if obj.__class__.__name__ == "Conv2d":
                ch_in = obj.weight.shape[1]
                ch_out = obj.weight.shape[0]
                kernel_size_x = obj.weight.shape[2]
                kernel_size_y = obj.weight.shape[3]
                iterations = _floor(int(input_shape[2]/obj.stride[0]), 2) * _floor(int(input_shape[3]/obj.stride[0]), 8)
                im2col = kernel_size_x * kernel_size_y * ch_in * 2
                matmul = _floor(ch_out, 4) * (5 + _floor(kernel_size_x * kernel_size_y * ch_in, 4) * (6 + 8) + 10)
                latency += iterations * (im2col + matmul)
                if obj.groups > 1:
                    # 1 MAC/cycle
                    latency = 4 * _floor(ch_out, 4) * int(input_shape[2]/obj.stride[0]) * int(input_shape[3]/obj.stride[0]) * kernel_size_x * kernel_size_y
            elif obj.__class__.__name__ == "Identity":
                latency += 0
            elif obj.__class__.__name__ == "Linear":
                ch_in = obj.weight.shape[1]
                ch_out = obj.weight.shape[0]
                latency += _floor(ch_in, 2) * _floor(ch_out, 4)
        import pdb;pdb.set_trace()
        self.layers_macs.append(latency)

        
def get_size_binarized_supernet(self) -> torch.Tensor:
    """Method that returns the number of weights for the module
    computed as a weighted sum of the number of weights of each layer.

    :return: number of weights of the module (weighted sum)
    :rtype: torch.Tensor
    """
    soft_alpha = nn.functional.softmax(self.alpha/self.softmax_temperature, dim=0)

    size = torch.tensor(0, dtype=torch.float32)
    for i in range(self.n_layers):
        var_size = torch.tensor(0, dtype=torch.float32)
        for pl in cast(List[PITModule], self._pit_layers[i]):
            var_size = var_size + pl.get_size()
        size = size + (soft_alpha[i] * (self.layers_sizes[i] + var_size))
    return size

def get_latency_binarized_supernet(self) -> torch.Tensor:
    """Method that computes the number of MAC operations for the module

    :return: the number of MACs
    :rtype: torch.Tensor
    """
    soft_alpha = nn.functional.softmax(self.alpha/self.softmax_temperature, dim=0)

    macs = torch.tensor(0, dtype=torch.float32)
    for i in range(self.n_layers):
        var_macs = torch.tensor(0, dtype=torch.float32)
        for pl in cast(List[PITModule], self._pit_layers[i]):
            var_macs = var_macs + pl.get_macs()
        macs = macs + (soft_alpha[i] * (self.layers_macs[i] + var_macs))
    return macs

def get_latency_conv2D_GAP8(self) -> torch.Tensor:
    iterations = _floor(self.out_height, 2) * _floor(self.out_width, 8)
    cin_mask = self.input_features_calculator.features_mask
    ch_in = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
    cout_mask = self.out_features_masker.theta
    ch_out = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
    im2col = self.kernel_size[0] * self.kernel_size[1] * ch_in * 2
    matmul = FloorSTE.apply(ch_out, 4) * (5 + FloorSTE.apply(self.kernel_size[0] * self.kernel_size[1] * ch_in, 4) * (6 + 8) + 10)
    latency = iterations * (im2col + matmul)
    if self.groups > 1:
        # 1 MAC/cycle
        latency = 4 * FloorSTE.apply(ch_out, 4) * self.out_height * self.out_width * self.kernel_size[0] * self.kernel_size[1]
    return latency

def get_latency_Linear_GAP8(self) -> torch.Tensor:
    cin_mask = self.input_features_calculator.features_mask
    ch_in = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
    # Compute actual integer number of output channels
    cout_mask = self.out_features_masker.theta
    ch_out = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
    # Finally compute cost
    latency = FloorSTE.apply(ch_in, 2) * FloorSTE.apply(ch_out, 4)
    return latency

def get_latency_total(self) -> torch.Tensor:
    latency = torch.tensor(0, dtype=torch.float32)
    # size = torch.tensor(0)
    for layer in self._target_layers:
        # size += layer.get_size()
        latency = latency + layer.get_latency()
    return latency

def get_latency_conv2D_Diana(self):
    pass

def get_latency_Linear_Diana(self):
    pass

def compute_layer_latency_Diana(self):
    pass

def get_memory_layer_constraints(self) -> torch.Tensor:
    """Computes the total number of parameters of all NAS-able layers
    using binary masks

    :return: the total number of parameters
    :rtype: torch.Tensor
    """
    size = torch.tensor(0, dtype=torch.float32)
    # size = torch.tensor(0)
    for layer in self._target_layers:
        # size += layer.get_size()
        size = size + layer.get_Lx_level_constraint()
    return size

def get_tot_mem_conv(self) -> torch.Tensor:
    # Compute actual integer number of input channels
    cin_mask = self.input_features_calculator.features_mask
    cin = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
    # Compute actual integer number of output channels
    cout_mask = self.out_features_masker.theta
    cout = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
    # Finally compute cost
    mem_in = (self.out_height * self.stride[0]) * (self.out_width * self.stride[1]) * cin
    mem_out = self.out_height * self.out_width * cout
    weights = cin * cout * self.kernel_size[0] * self.kernel_size[1]
    if self.groups == 1:
        weights = cin * cout * self.kernel_size[0] * self.kernel_size[1]
    else:
        weights = cout * self.kernel_size[0] * self.kernel_size[1]
    return (mem_in + mem_out + weights ) #- self.TARGET

def get_tot_mem_linear(self) -> torch.Tensor:
    # Compute actual integer number of input channels
    cin_mask = self.input_features_calculator.features_mask
    cin = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
    # Compute actual integer number of output channels
    cout_mask = self.out_features_masker.theta
    cout = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
    # Finally compute cost
    mem_in = cin
    mem_out = cout
    weights = cin * cout
    return (mem_in + mem_out + weights ) #- self.TARGET


def get_individual_mem_conv(self) -> torch.Tensor:
    # Compute actual integer number of input channels
    cin_mask = self.input_features_calculator.features_mask
    cin = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
    # Compute actual integer number of output channels
    cout_mask = self.out_features_masker.theta
    cout = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
    # Finally compute cost
    mem_in = (self.out_height * self.stride[0]) * (self.out_width * self.stride[1]) * cin
    mem_out = self.out_height * self.out_width * cout
    weights = cin * cout * self.kernel_size[0] * self.kernel_size[1]
    if self.groups == 1:
        weights = cin * cout * self.kernel_size[0] * self.kernel_size[1]
    else:
        weights = cout * self.kernel_size[0] * self.kernel_size[1]
    return mem_in, mem_out, weights

def get_individual_mem_linear(self) -> torch.Tensor:
    # Compute actual integer number of input channels
    cin_mask = self.input_features_calculator.features_mask
    cin = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
    # Compute actual integer number of output channels
    cout_mask = self.out_features_masker.theta
    cout = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
    # Finally compute cost
    mem_in = cin
    mem_out = cout
    weights = cin * cout
    return mem_in, mem_out, weights

def get_Lx_level_constraint_conv(self) -> torch.Tensor:
    # Compute actual integer number of input channels
    cin_mask = self.input_features_calculator.features_mask
    cin = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
    # Compute actual integer number of output channels
    cout_mask = self.out_features_masker.theta
    cout = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
    # Finally compute cost
    mem_in = (self.out_height * self.stride[0]) * (self.out_width * self.stride[1]) * cin
    mem_out = self.out_height * self.out_width * cout
    weights = cin * cout * self.kernel_size[0] * self.kernel_size[1]
    if self.groups == 1:
        weights = cin * cout * self.kernel_size[0] * self.kernel_size[1]
    else:
        weights = cout * self.kernel_size[0] * self.kernel_size[1]
    return torch.max(( mem_in + mem_out + weights - self.TARGET), torch.FloatTensor([0]).to("cuda"))[0]

def get_Lx_level_constraint_linear(self) -> torch.Tensor:
    # Compute actual integer number of input channels
    cin_mask = self.input_features_calculator.features_mask
    cin = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
    # Compute actual integer number of output channels
    cout_mask = self.out_features_masker.theta
    cout = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
    # Finally compute cost
    mem_in = cin
    mem_out = cout
    weights = cin * cout
    return torch.max(( mem_in + mem_out + weights - self.TARGET), torch.FloatTensor([0]).to("cuda"))[0]