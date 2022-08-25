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
import copy

def track_ch(name, model):
    def hook(module, input, output):
        model.alive_ch[name] = output[1].clone().detach().cpu().numpy().sum()
    return hook

def track_complexity(name, model):
    def hook(module, input, output):
        model.size_dict[name] = output[2]
        model.ops_dict[name] = output[3]
    return hook

def register_hook(model, conv_func, *args):
    registered_model = copy.deepcopy(model)
    for name, module in registered_model.named_modules():
        if isinstance(module, conv_func):
            for arg in args:
                module.register_forward_hook(arg(name, registered_model))
    return registered_model
