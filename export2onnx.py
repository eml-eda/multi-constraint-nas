import argparse
import copy
import json

import torch
import torch.nn as nn

import models as models

parser = argparse.ArgumentParser(description='Export model onnx for deployment')
parser.add_argument('arch', type=str, help='Architecture name')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    # Build Model
    model = models.__dict__[args.arch]()
    if 'searchable' in str(args.arch):
        model_name = str(args.arch).split('searchable')[1].split('_')[1]
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
    torch.onnx.export(model, rnd_input, str(args.arch)+'.onnx', verbose=True, do_constant_folding=False, training=2)