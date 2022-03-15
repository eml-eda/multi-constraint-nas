import argparse
import copy

import torch

import models as models

parser = argparse.ArgumentParser(description='Compute Complexities')
parser.add_argument('arch', type=str, help='Architecture name')
parser.add_argument('--pretrained-model', type=str, default=None, help='Pretrained model path')

args = parser.parse_args()
print(args)

# Random input
model_name = str(args.arch).split('searchable')[1].split('_')[1]
if model_name == 'dscnn':
    rnd_input = torch.randn(2, 1, 49, 10)
elif model_name == 'resnet8':
    rnd_input = torch.randn(2, 3, 32, 32)
else:
    raise ValueError(f'Model {model_name} not supported')

# Get initial model complexities
model = models.__dict__[args.arch]()
# Dummy forward pass
with torch.no_grad():
    out = model(rnd_input)
size_i = sum(model.size_dict.values())#.clone().detach().cpu().numpy()
ops_i = sum(model.ops_dict.values())#.clone().detach().cpu().numpy()
print(f"Initial size: {size_i:.3e} params\tInitial ops: {ops_i:.3e} OPs")
alive_ch_i = copy.deepcopy(model.alive_ch)

# Load pretrained model if specified
if args.pretrained_model is not None:
    model.load_state_dict(torch.load(args.pretrained_model)) 

# Dummy forward pass
with torch.no_grad():
    model(rnd_input)

# Compute and print final size and ops
size_f = sum(model.size_dict.values())#.clone().detach().cpu().numpy()
ops_f = sum(model.ops_dict.values())#.clone().detach().cpu().numpy()
print(f"Final size: {size_f:.3e}/{size_i:.3e} parameters\tFinal ops: {ops_f:.3e}/{ops_i:.3e} OPs")
# Print learned alive channels
for k, v in model.alive_ch.items():
    print(f"{k}:\t{int(v)+1}/{int(alive_ch_i[k])+1} channels")