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
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import hook_func as hf
from models import search_module as sm
import utils

__all__ = [
    'plain_cnn', 'searchable_cnn'
]

def conv(conv_func, inplane, outplane, kernel_size, stride=1, bias=False, **kwargs):
    return conv_func(inplane, outplane, kernel_size=kernel_size, stride=stride,
                        padding=0, bias=bias, groups=1, **kwargs)

class SimpleCNN(nn.Module):
    def __init__(self, conv_func, **kwargs):
        super().__init__()
        self.conv1 = conv_func(1, 32, 3, 1)
        self.conv2 = conv_func(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = conv_func(9216, 128, 1)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x.view(x.shape[0], x.shape[1], 1, 1))
        x = F.relu(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class SearchableSimpleCNN(nn.Module):
    def __init__(self, conv_func, **kwargs):
        super().__init__()
        self.conv1 = conv(conv_func, 1, 32, 3, bias=True)
        self.conv2 = conv(conv_func, 32, 64, 3, bias=True)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = conv(conv_func, 9216, 128, 1, bias=True)
        self.fc2 = nn.Linear(128, 10)
        # Dictionaries with alive ch for each searched layer
        self.alive_ch = {}
        # Dictionaries where complexities are stored
        self.size_dict = {}
        self.ops_dict = {}

    def forward(self, x):
        x, alpha_1, size, ops = self.conv1(x)
        x = F.relu(x)
        x, alpha_2, size, ops = self.conv2(x, alpha_1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x, alpha_3, size, ops = self.fc1(x, alpha_2)
        x = F.relu(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def plain_cnn(**kwargs):
    return SimpleCNN(nn.Conv2d, **kwargs)

def searchable_cnn(found_model=None, **kwargs):
    model = SearchableSimpleCNN(sm.SearchableConv2d, **kwargs)
    ft = kwargs.pop('ft', False)
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