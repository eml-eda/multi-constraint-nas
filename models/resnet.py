#*----------------------------------------------------------------------------*
#* Copyright (C) 2022 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Matteo Risso <matteo.risso@polito.it>                             *
#*----------------------------------------------------------------------------*
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import hook_func as hf
from models import search_module as sm
import utils

__all__ = [
    'plain_resnet8', 'searchable_resnet8', 'learned_resnet8',
]

# Wrapping conv with conv_func
def conv3x3(conv_func, in_planes, out_planes, stride=1, groups=1, **kwargs):
    "3x3 convolution with padding"
    return conv_func(in_planes, out_planes, kernel_size=3, groups=groups, stride=stride,
                        padding=1, bias=False, **kwargs)
def res_conv(conv_func, in_planes, out_planes, alpha=None, stride=1, groups=1, **kwargs):
    "1x1 residual convolution with padding"
    return conv_func(in_planes, out_planes, alpha=alpha, kernel_size=1, groups=groups, stride=stride,
                        padding=0, bias=False, **kwargs)

# Wrapping fc with conv_func
def fc(conv_func, in_planes, out_planes, stride=1, groups=1, **kwargs):
    "fc mapped to conv"
    return conv_func(in_planes, out_planes, kernel_size=1, groups = groups, stride=stride,
                     padding=0, bias=False, **kwargs)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class ResNet8(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Resnet v1 parameters
        self.input_shape = [3, 32, 32]  # default size for cifar10
        self.num_classes = 10  # default class number for cifar10
        self.num_filters = 16  # this should be 64 for an official resnet model

        # Resnet v1 layers

        # First stack
        self.inputblock = ConvBlock(in_channels=3, out_channels=16,
                                    kernel_size=3, stride=1, padding=1)
        self.convblock1 = ConvBlock(in_channels=16, out_channels=16,
                                    kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # Second stack
        self.convblock2 = ConvBlock(in_channels=16, out_channels=32,
                                    kernel_size=3, stride=2, padding=1)
        self.conv2y = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2y.weight)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2x = nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0)
        nn.init.kaiming_normal_(self.conv2x.weight)

        # Third stack
        self.convblock3 = ConvBlock(in_channels=32, out_channels=64,
                                    kernel_size=3, stride=2, padding=1)
        self.conv3y = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv3y.weight)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3x = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)
        nn.init.kaiming_normal_(self.conv3x.weight)

        self.avgpool = torch.nn.AvgPool2d(8)

        self.out = nn.Linear(64, 10)
        nn.init.kaiming_normal_(self.out.weight)

    def forward(self, input):
        # Input layer
        x = self.inputblock(input)  # [32, 32, 16]

        # First stack
        y = self.convblock1(x)      # [32, 32, 16]
        y = self.conv1(y)
        y = self.bn1(y)
        x = torch.add(x, y)         # [32, 32, 16]
        x = self.relu(x)

        # Second stack
        y = self.convblock2(x)      # [16, 16, 32]
        y = self.conv2y(y)
        y = self.bn2(y)
        x = self.conv2x(x)          # [16, 16, 32]
        x = torch.add(x, y)         # [16, 16, 32]
        x = self.relu(x)

        # Third stack
        y = self.convblock3(x)      # [8, 8, 64]
        y = self.conv3y(y)
        y = self.bn3(y)
        x = self.conv3x(x)          # [8, 8, 64]
        x = torch.add(x, y)         # [8, 8, 64]
        x = self.relu(x)

        x = self.avgpool(x)         # [1, 1, 64]
        # x = torch.squeeze(x)        # [64]
        x = torch.flatten(x, 1)
        x = self.out(x)             # [10]

        return x


class LearnedResNet8(nn.Module):
    def __init__(self, conv_func, learned_ch, input_size=32, num_classes=10, **kwargs):
        self.inplanes = 16
        self.conv_func = conv_func
        super().__init__()
        # Input layer
        self.conv0 = conv_func(3, learned_ch[0], 3, 1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(learned_ch[0])
        
        # First stack - bb0
        self.bb0_conv1 = conv_func(learned_ch[0], learned_ch[1], 3, 1, padding=1, bias=False)
        self.bb0_bn1 = nn.BatchNorm2d(learned_ch[1])
        self.bb0_conv2 = conv_func(learned_ch[1], learned_ch[2], 3, 1, padding=1, bias=False)
        self.bb0_bn2 = nn.BatchNorm2d(learned_ch[2])
        self.bb0_res = conv_func(learned_ch[0], learned_ch[2], 1, 1, padding=0, bias=False) 
        
        # Second stack - bb1
        self.bb1_conv1 = conv_func(learned_ch[2], learned_ch[3], 3, 2, padding=1, bias=False)
        self.bb1_bn1 = nn.BatchNorm2d(learned_ch[3])
        self.bb1_conv2 = conv_func(learned_ch[3], learned_ch[4], 3, 1, padding=1, bias=False)
        self.bb1_bn2 = nn.BatchNorm2d(learned_ch[4])
        self.bb1_res = conv_func(learned_ch[2], learned_ch[4], 1, 2, padding=0, bias=False) 
        
        # Third stack - bb2
        self.bb2_conv1 = conv_func(learned_ch[4], learned_ch[5], 3, 2, padding=1, bias=False)
        self.bb2_bn1 = nn.BatchNorm2d(learned_ch[5])
        self.bb2_conv2 = conv_func(learned_ch[5], learned_ch[6], 3, 1, padding=1, bias=False)
        self.bb2_bn2 = nn.BatchNorm2d(learned_ch[6])
        self.bb2_res = conv_func(learned_ch[4], learned_ch[6], 1, 2, padding=0, bias=False)
        
        # Final classifier
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = conv_func(learned_ch[6], num_classes, 1, 1)

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
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)

        # First stack - bb0
        x0 = x
        x0_res = self.bb0_res(x0)
        x = self.bb0_conv1(x0)
        x = self.bb0_bn1(x)
        x = F.relu(x)
        x = self.bb0_conv2(x)
        x = self.bb0_bn2(x)
        x = x + x0_res
        x = F.relu(x)

        # Second stack - bb1
        x1 = x
        x1_res = self.bb1_res(x1)
        x = self.bb1_conv1(x1)
        x = self.bb1_bn1(x)
        x = F.relu(x)
        x = self.bb1_conv2(x)
        x = self.bb1_bn2(x)
        x = x + x1_res
        x = F.relu(x)

        # Third stack - bb2
        x2 = x
        x2_res = self.bb2_res(x2)
        x = self.bb2_conv1(x2)
        x = self.bb2_bn1(x)
        x = F.relu(x)
        x = self.bb2_conv2(x)
        x = self.bb2_bn2(x)
        x = x + x2_res
        x = F.relu(x)

        # Final classifier
        x = self.avgpool(x) 
        x = self.fc(x)

        return x

class SearchableResNet8(nn.Module):
    def __init__(self, conv_func, input_size=32, num_classes=10, **kwargs):
        self.inplanes = 16
        self.conv_func = conv_func
        super().__init__()
        # Input layer
        self.conv0 = conv3x3(conv_func, 3, 16, stride=1, groups=1, **kwargs)
        self.bn0 = nn.BatchNorm2d(16)
        
        # First stack - bb0
        self.bb0_conv1 = conv3x3(conv_func, 16, 16, stride=1, groups=1, **kwargs)
        self.bb0_bn1 = nn.BatchNorm2d(16)
        self.bb0_conv2 = conv3x3(conv_func, 16, 16, stride=1, groups=1, **kwargs)
        self.bb0_bn2 = nn.BatchNorm2d(16)
        
        # Second stack - bb1
        self.bb1_conv1 = conv3x3(conv_func, 16, 32, stride=2, groups=1, **kwargs)
        self.bb1_bn1 = nn.BatchNorm2d(32)
        self.bb1_conv2 = conv3x3(conv_func, 32, 32, stride=1, groups=1, **kwargs)
        self.bb1_bn2 = nn.BatchNorm2d(32)
        self.bb1_res = res_conv(conv_func, 16, 32, alpha=self.bb1_conv2.alpha, stride=2, groups=1, **kwargs) 
        
        # Third stack - bb2
        self.bb2_conv1 = conv3x3(conv_func, 32, 64, stride=2, groups=1, **kwargs)
        self.bb2_bn1 = nn.BatchNorm2d(64)
        self.bb2_conv2 = conv3x3(conv_func, 64, 64, stride=1, groups=1, **kwargs)
        self.bb2_bn2 = nn.BatchNorm2d(64)
        self.bb2_res = res_conv(conv_func, 32, 64, alpha=self.bb2_conv2.alpha, stride=2, groups=1, **kwargs)
        
        # Final classifier
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = fc(conv_func, 64, num_classes, **kwargs)

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
        x, alpha_0, size, ops = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)

        # First stack - bb0
        x0 = x
        x, alpha_0_1, size, ops = self.bb0_conv1(x0, alpha_0)
        x = self.bb0_bn1(x)
        x = F.relu(x)
        x, alpha_0_2, size, ops = self.bb0_conv2(x, alpha_0_1)
        x = self.bb0_bn2(x)
        x = x + x0
        x = F.relu(x)

        # Second stack - bb1
        x1 = x
        x1_res, alpha_1_res, size, ops = self.bb1_res(x1, alpha_0_2)
        x, alpha_1_1, size, ops = self.bb1_conv1(x1, alpha_0_2)
        x = self.bb1_bn1(x)
        x = F.relu(x)
        x, alpha_1_2, size, ops = self.bb1_conv2(x, alpha_1_1)
        x = self.bb1_bn2(x)
        x = x + x1_res
        x = F.relu(x)

        # Third stack - bb2
        x2 = x
        x2_res, alpha_2_res, size, ops = self.bb2_res(x2, alpha_1_2)
        x, alpha_2_1, size, ops = self.bb2_conv1(x2, alpha_1_2)
        x = self.bb2_bn1(x)
        x = F.relu(x)
        x, alpha_2_2, size, ops = self.bb2_conv2(x, alpha_2_1)
        x = self.bb2_bn2(x)
        x = x + x2_res
        x = F.relu(x)

        # Final classifier
        x = self.avgpool(x) 
        x, alpha_fc, size, ops = self.fc(x, alpha_2_2)

        return x[:, :, 0, 0]

def plain_resnet8(**kwargs):
    return ResNet8(**kwargs)

def learned_resnet8(learned_ch, **kwargs):
    return LearnedResNet8(nn.Conv2d, learned_ch, **kwargs)

def searchable_resnet8(found_model=None, **kwargs):
    ft = kwargs.pop('ft', False)
    model = SearchableResNet8(sm.SearchableConv2d, **kwargs)
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