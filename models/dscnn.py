import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import hook_func as hf
from models import search_module as sm
import utils

__all__ = [
    'plain_dscnn', 'searchable_dscnn'
]

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
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv1x1(conv_func, inplanes, planes, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes)

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
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv1x1(conv_func, inplanes, planes, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x, alpha_prev=None):
        out, alpha_0, size, ops = self.conv0(x, alpha_prev)
        out = self.bn0(out)
        out = F.relu(out)
        out, alpha_1, size, ops = self.conv1(out, alpha_0)
        out = self.bn1(out)
        out = F.relu(out)
        return out, alpha_1

class DsCNN(nn.Module):
    def __init__(self, conv_func, input_size=(49,10), num_classes=12, **kwargs):
        self.conv_func = conv_func
        super().__init__()
        # Input layer
        self.input_layer = conv_func(1, 64, kernel_size=(10,4), stride=2, padding=(5,1), bias=False, groups=1, **kwargs)
        self.bn = nn.BatchNorm2d(64)
        self.dpout0 = nn.Dropout(0.2)

        # Backbone
        self.bb_1 = BasicBlock(conv_func, 64, 64, 1, **kwargs)
        self.bb_2 = BasicBlock(conv_func, 64, 64, 1, **kwargs)
        self.bb_3 = BasicBlock(conv_func, 64, 64, 1, **kwargs)
        self.bb_4 = BasicBlock(conv_func, 64, 64, 1, **kwargs)
        self.dpout1 = nn.Dropout(0.4)
        
        # Final classifier
        self.pool = nn.AvgPool2d((int(input_size[0]/2), int(input_size[1]/2)))
        self.fc = fc(conv_func, 64, num_classes, **kwargs)

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
        x = self.dpout0(x)

        # Backbone
        x = self.bb_1(x)
        x = self.bb_2(x)
        x = self.bb_3(x)
        x = self.bb_4(x)
        x = self.dpout1(x)

        # Final classifier
        x = self.pool(x)
        x = self.fc(x)

        return x[:, :, 0, 0]

class SearchableDsCNN(nn.Module):
    def __init__(self, conv_func, input_size=(49,10), num_classes=12, **kwargs):
        self.conv_func = conv_func
        super().__init__()
        # Input layer
        self.input_layer = conv_func(1, 64, kernel_size=(10,4), stride=2, padding=(5,1), bias=False, groups=1, **kwargs)
        self.bn = nn.BatchNorm2d(64)
        self.dpout0 = nn.Dropout(0.2)

        # Backbone
        self.bb_1 = SearchableBasicBlock(conv_func, 64, 64, 1, **kwargs)
        self.bb_2 = SearchableBasicBlock(conv_func, 64, 64, 1, **kwargs)
        self.bb_3 = SearchableBasicBlock(conv_func, 64, 64, 1, **kwargs)
        self.bb_4 = SearchableBasicBlock(conv_func, 64, 64, 1, **kwargs)
        self.dpout1 = nn.Dropout(0.4)
        
        # Final classifier
        self.pool = nn.AvgPool2d((int(input_size[0]/2), int(input_size[1]/2)))
        self.fc = fc(conv_func, 64, num_classes, **kwargs)

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
        x = self.dpout0(x)

        # Backbone
        x, alpha_1, size, ops = self.bb_1(x, alpha_0)
        x, alpha_2, size, ops = self.bb_2(x, alpha_1)
        x, alpha_3, size, ops = self.bb_3(x, alpha_2)
        x, alpha_4, size, ops = self.bb_4(x, alpha_3)
        x = self.dpout1(x)

        # Final classifier
        x = self.pool(x)
        x, alpha_fc, size, ops = self.fc(x, alpha_4)

        return x[:, :, 0, 0]

def plain_dscnn(**kwargs):
    return DsCNN(nn.Conv2d, **kwargs)

def searchable_dscnn(found_model=None, **kwargs):
    ft = kwargs.pop('ft', False)
    model = SearchableDsCNN(sm.SearchableConv2d, **kwargs)
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