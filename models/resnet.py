import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import hook_func as hf
from models import search_module as sm
import utils

__all__ = [
    'plain_resnet8', 'searchable_resnet8'
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

class ResNet8(nn.Module):
    def __init__(self, conv_func, input_size=32, num_classes=10, **kwargs):
        self.inplanes = 16
        self.conv_func = conv_func
        super().__init__()
        # Input layer
        self.conv0 = conv_func(3, 16, 3, 1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)
        
        # First stack - bb0
        self.bb0_conv1 = conv_func(16, 16, 3, 1, padding=1, bias=False)
        self.bb0_bn1 = nn.BatchNorm2d(16)
        self.bb0_conv2 = conv_func(16, 16, 3, 1, padding=1, bias=False)
        self.bb0_bn2 = nn.BatchNorm2d(16)
        
        # Second stack - bb1
        self.bb1_conv1 = conv_func(16, 32, 3, 2, padding=1, bias=False)
        self.bb1_bn1 = nn.BatchNorm2d(32)
        self.bb1_conv2 = conv_func(32, 32, 3, 1, padding=1, bias=False)
        self.bb1_bn2 = nn.BatchNorm2d(32)
        self.bb1_res = conv_func(16, 32, 1, 2, padding=0, bias=False) 
        
        # Third stack - bb2
        self.bb2_conv1 = conv_func(32, 64, 3, 2, padding=1, bias=False)
        self.bb2_bn1 = nn.BatchNorm2d(64)
        self.bb2_conv2 = conv_func(64, 64, 3, 1, padding=1, bias=False)
        self.bb2_bn2 = nn.BatchNorm2d(64)
        self.bb2_res = conv_func(32, 64, 1, 2, padding=0, bias=False)
        
        # Final classifier
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = conv_func(64, num_classes, 1, 1)

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
        x = self.bb0_conv1(x0)
        x = self.bb0_bn1(x)
        x = F.relu(x)
        x = self.bb0_conv2(x)
        x = self.bb0_bn2(x)
        x = x + x0
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
        x = self.fc(x)[:, :, 0, 0]

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
    return ResNet8(nn.Conv2d, **kwargs)

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