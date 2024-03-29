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
import argparse
import copy
import json

import torch
import torch.nn as nn

import models as models

parser = argparse.ArgumentParser(description='Export model onnx for deployment')
parser.add_argument('arch', type=str, help='Architecture name')
parser.add_argument('--output-file', type=str, help='output file name')
parser.add_argument('--learned-ch', nargs='+', default=None, type=int, help='List of learned channels')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    # Build Model
    if args.learned_ch is None:
        model = models.__dict__[args.arch]()
    else:
        model = models.__dict__[args.arch](args.learned_ch)

    if 'searchable' in str(args.arch):
        model_name = str(args.arch).split('searchable')[1].split('_')[1]
    if 'learned' in str(args.arch):
        model_name = str(args.arch).split('learned')[1].split('_')[1]
    elif 'plain' in str(args.arch):
        model_name = str(args.arch).split('plain')[1].split('_')[1]

    # Random input
    if model_name == 'dscnn':
        rnd_input = torch.randn(2, 1, 49, 10)
    elif model_name == 'resnet8':
        rnd_input = torch.randn(2, 3, 32, 32)
    elif model_name == 'mobilenetv1':
        rnd_input = torch.randn(1, 3, 96, 96)
    else:
        raise ValueError(f'Model {model_name} not supported')

    # Export onnx
    torch.onnx.export(model, rnd_input, str(args.output_file), verbose=True, do_constant_folding=False, training=2)