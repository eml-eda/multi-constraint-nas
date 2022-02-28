import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import hook_func as hf
from models import search_module as sm
import utils

__all__ = [
    'plain_mobilenetv1', 'searchable_mobilenetv1'
]

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

# Wrapping pointwise conv with conv_func
def conv1x1(conv_func, in_planes, out_planes, stride=1, **kwargs):
    "1x1 convolution with padding"
    return conv_func(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False, groups = 1, **kwargs)

# Wrapping depthwise conv with conv_func
def dw3x3(conv_func, in_planes, out_planes, stride=1, **kwargs):
    "3x3 convolution dw with padding"
    return conv_func(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=in_planes, **kwargs)

# Wrapping fc with conv_func
def fc(conv_func, in_planes, out_planes, stride=1, groups=1, **kwargs):
    "fc mapped to conv"
    return conv_func(in_planes, out_planes, kernel_size=1, groups=groups, stride=stride,
                     padding=0, bias=True, **kwargs)

class BasicBlock(nn.Module):
    def __init__(self, conv_func, inplanes, planes, stride=1, **kwargs):
        super().__init__()
        self.conv0 = dw3x3(conv_func, inplanes, inplanes, stride=stride, **kwargs)
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        self.conv1 = conv1x1(conv_func, inplanes, planes, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        return out

class SearchableBasicBlock(nn.Module):
    def __init__(self, conv_func, inplanes, planes, stride=1, **kwargs):
        super().__init__()
        self.conv0 = dw3x3(conv_func, inplanes, inplanes, stride=stride, **kwargs)
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        self.conv1 = conv1x1(conv_func, inplanes, planes, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)

    def forward(self, x, alpha_prev=None):
        out, alpha_0, size, ops = self.conv0(x, alpha_prev)
        out = self.bn0(out)
        out = F.relu(out)
        out, alpha_1, size, ops = self.conv1(out, alpha_0)
        out = self.bn1(out)
        out = F.relu(out)
        return out, alpha_1

class MobileNetV1(nn.Module):
    def __init__(self, conv_func, input_size=96, num_classes=2, width_mult=.25, **kwargs):
        self.conv_func = conv_func
        super().__init__()
        # Input layer
        self.input_layer = conv_func(3, make_divisible(32*width_mult), kernel_size=3, stride=2, padding=1, bias=False, groups=1, **kwargs)
        self.bn = nn.BatchNorm2d(make_divisible(32*width_mult))

        # Backbone
        self.bb_1 = BasicBlock(conv_func, make_divisible(32*width_mult), make_divisible(64*width_mult), 1, **kwargs)
        self.bb_2 = BasicBlock(conv_func, make_divisible(64*width_mult), make_divisible(128*width_mult), 2, **kwargs)
        self.bb_3 = BasicBlock(conv_func, make_divisible(128*width_mult), make_divisible(128*width_mult), 1, **kwargs)
        self.bb_4 = BasicBlock(conv_func, make_divisible(128*width_mult), make_divisible(256*width_mult), 2, **kwargs)
        self.bb_5 = BasicBlock(conv_func, make_divisible(256*width_mult), make_divisible(256*width_mult), 1, **kwargs)
        self.bb_6 = BasicBlock(conv_func, make_divisible(256*width_mult), make_divisible(512*width_mult), 2, **kwargs)
        self.bb_7 = BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_8 = BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_9 = BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_10 = BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_11 = BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_12 = BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(1024*width_mult), 2, **kwargs)
        self.bb_13 = BasicBlock(conv_func, make_divisible(1024*width_mult), make_divisible(1024*width_mult), 1, **kwargs)
        
        # Final classifier
        self.pool = nn.AvgPool2d(int(input_size/(2**5)))
        self.fc = fc(conv_func, make_divisible(1024*width_mult), num_classes, **kwargs)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        # Input Layer
        x = self.input_layer(x)
        x = self.bn(x)
        x = F.relu(x)

        # Backbone
        x = self.bb_1(x)
        x = self.bb_2(x)
        x = self.bb_3(x)
        x = self.bb_4(x)
        x = self.bb_5(x)
        x = self.bb_6(x)
        x = self.bb_7(x)
        x = self.bb_8(x)
        x = self.bb_9(x)
        x = self.bb_10(x)
        x = self.bb_11(x)
        x = self.bb_12(x)
        x = self.bb_13(x)

        # Final classifier
        x = self.pool(x)
        x = self.fc(x)

        return x[:, :, 0, 0]

class SearchableMobileNetV1(nn.Module):
    def __init__(self, conv_func, input_size=96, num_classes=2, width_mult=.25, **kwargs):
        self.conv_func = conv_func
        super().__init__()
        # Input layer
        self.input_layer = conv_func(3, make_divisible(32*width_mult), kernel_size=3, stride=2, padding=1, bias=False, groups=1, **kwargs)
        self.bn = nn.BatchNorm2d(make_divisible(32*width_mult))

        # Backbone
        self.bb_1 = SearchableBasicBlock(conv_func, make_divisible(32*width_mult), make_divisible(64*width_mult), 1, **kwargs)
        self.bb_2 = SearchableBasicBlock(conv_func, make_divisible(64*width_mult), make_divisible(128*width_mult), 2, **kwargs)
        self.bb_3 = SearchableBasicBlock(conv_func, make_divisible(128*width_mult), make_divisible(128*width_mult), 1, **kwargs)
        self.bb_4 = SearchableBasicBlock(conv_func, make_divisible(128*width_mult), make_divisible(256*width_mult), 2, **kwargs)
        self.bb_5 = SearchableBasicBlock(conv_func, make_divisible(256*width_mult), make_divisible(256*width_mult), 1, **kwargs)
        self.bb_6 = SearchableBasicBlock(conv_func, make_divisible(256*width_mult), make_divisible(512*width_mult), 2, **kwargs)
        self.bb_7 = SearchableBasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_8 = SearchableBasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_9 = SearchableBasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_10 = SearchableBasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_11 = SearchableBasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_12 = SearchableBasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(1024*width_mult), 2, **kwargs)
        self.bb_13 = SearchableBasicBlock(conv_func, make_divisible(1024*width_mult), make_divisible(1024*width_mult), 1, **kwargs)
        
        # Final classifier
        self.pool = nn.AvgPool2d(int(input_size/(2**5)))
        self.fc = fc(conv_func, make_divisible(1024*width_mult), num_classes, **kwargs)

        # Dictionaries with alive ch for each searched layer
        self.alive_ch = {}
        # Dictionaries where complexities are stored
        self.size_dict = {}
        self.ops_dict = {}

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        # Input Layer
        x, alpha_0, size, ops = self.input_layer(x)
        x = self.bn(x)
        x = F.relu(x)

        # Backbone
        x, alpha_1 = self.bb_1(x, alpha_0)
        x, alpha_2 = self.bb_2(x, alpha_1)
        x, alpha_3 = self.bb_3(x, alpha_2)
        x, alpha_4 = self.bb_4(x, alpha_3)
        x, alpha_5 = self.bb_5(x, alpha_4)
        x, alpha_6 = self.bb_6(x, alpha_5)
        x, alpha_7 = self.bb_7(x, alpha_6)
        x, alpha_8 = self.bb_8(x, alpha_7)
        x, alpha_9 = self.bb_9(x, alpha_8)
        x, alpha_10 = self.bb_10(x, alpha_9)
        x, alpha_11 = self.bb_11(x, alpha_10)
        x, alpha_12 = self.bb_12(x, alpha_11)
        x, alpha_13 = self.bb_13(x, alpha_12)

        # Final classifier
        x = self.pool(x)
        x, alpha_fc, size, ops = self.fc(x, alpha_13)

        return x[:, :, 0, 0]

def plain_mobilenetv1(**kwargs):
    return MobileNetV1(nn.Conv2d, **kwargs)

def searchable_mobilenetv1(found_model=None, **kwargs):
    ft = kwargs.pop('ft', False)
    model = SearchableMobileNetV1(sm.SearchableConv2d, **kwargs)
    if not ft:
        # Register model with hooks tracking complexities
        registered_model = hf.register_hook(model, sm.SearchableConv2d, hf.track_complexity, hf.track_ch)
        return registered_model
    else:
        # Freeze searchable parameters
        freezed_model = utils.freeze_model(model, sm.SearchableConv2d)
        if found_model is not None: # Load end-of-search state-dict
            freezed_model.load_state_dict(torch.load(found_model))
            return freezed_model
        else: # Perform training from scratch
            return freezed_model