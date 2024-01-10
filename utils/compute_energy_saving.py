import argparse
import pathlib
from typing import Dict
import os 

from pytorch_model_summary import summary
from torchinfo import summary as summ
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from plinio.methods import PIT
from plinio.methods import PITSuperNet
from plinio.methods.pit.nn import PITConv2d, PITLinear
from plinio.methods.pit_supernet.nn import PITSuperNetCombiner
import pytorch_benchmarks.tiny_imagenet as icl
from pytorch_benchmarks.utils import AverageMeter, seed_all, accuracy, CheckPoint, EarlyStopping
from hardware_model import get_memory_layer_constraints, get_Lx_level_constraint_conv, get_Lx_level_constraint_linear, get_individual_mem_conv, get_individual_mem_linear

from utils import evaluate, train_one_epoch_finetuning, train_one_epoch, train_one_epoch_finetuning_sizeloss
from models import ResNet8PITSN

def main(args):
    data_dir = args.data_dir
    N_EPOCHS = args.epochs
    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)
    # Ensure determinstic execution
    seed_all(seed=14)
    # Get the Data
    datasets2 = icl.get_data(inp_res=64)
    dataloaders2 = icl.build_dataloaders(datasets2)
    train_dl, val_dl, test_dl_all = dataloaders2
    def load(model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    config = {'pretrained': False, 'std_head': False}
    model = icl.get_reference_model('resnet_18', model_config=config).to(device)
    input_example = torch.unsqueeze(datasets2[0][0][0], 0).to(device)
    input_shape = datasets2[0][0][0].numpy().shape
    criterion = icl.get_default_criterion()
    if args.model == "PIT":
        pit_model = PIT(model, input_shape=input_shape).to(device)
    else:
        PITSuperNet.get_size_binarized = PITSuperNet.get_size
        PITSuperNet.get_macs_binarized = PITSuperNet.get_macs
        pit_model = ResNet8PITSN(gumbel = True).to(device)
        pit_model = PITSuperNet(pit_model, input_shape=input_shape, autoconvert_layers = False).to(device)
    log = 0
    for file in os.listdir("search_checkpoints"):
        if f"ck_tiny_opt_{args.model}_max_targets_mem_constraint+lat_constraint_increasing_size_{args.size_target}_lat_{args.latency_target}_" in file:
            if int(file.split(".")[-2].split("_")[-1]) > log:
                log = int(file.split(".")[-2].split("_")[-1])
    if log < 100:
        pit_model=load(pit_model, f'search_checkpoints/ck_tiny_opt_{args.model}_max_targets_mem_constraint+lat_constraint_increasing_size_{args.size_target}_lat_{args.latency_target}_0{log}.pt')
    else:
        pit_model=load(pit_model, f'search_checkpoints/ck_tiny_opt_{args.model}_max_targets_mem_constraint+lat_constraint_increasing_size_{args.size_target}_lat_{args.latency_target}_{log}.pt')
    
    print("final model size:", pit_model.get_size_binarized())
    print("final model MACs:", pit_model.get_macs_binarized())
    # Convert pit model into pytorch model
    exported_model = pit_model.arch_export()
    exported_model = exported_model.to(device)
    print(summary(exported_model, input_example, show_input=False, show_hierarchical=True))
    log = 0
    for file in os.listdir("finetuning_checkpoints"):
        if f"ck_tiny_opt_{args.model}_max_targets_mem_constraint+lat_constraint_increasing_size_{args.size_target}_lat_{args.latency_target}_" in file:
            if int(file.split(".")[-2].split("_")[-1]) > log:
                log = int(file.split(".")[-2].split("_")[-1])
    if log < 100:
        exported_model=load(exported_model, f'finetuning_checkpoints/ck_tiny_opt_{args.model}_max_targets_mem_constraint+lat_constraint_increasing_size_{args.size_target}_lat_{args.latency_target}_0{log}.pt')
    else:
        exported_model=load(exported_model, f'finetuning_checkpoints/ck_tiny_opt_{args.model}_max_targets_mem_constraint+lat_constraint_increasing_size_{args.size_target}_lat_{args.latency_target}_{log}.pt')
    PITConv2d.get_tot_mem = get_individual_mem_conv
    PITLinear.get_tot_mem = get_individual_mem_linear
    PITConv2d.get_Lx_level_constraint = get_Lx_level_constraint_conv
    PITLinear.get_Lx_level_constraint = get_Lx_level_constraint_linear
    PIT.get_memory_layer_constraints = get_memory_layer_constraints
    pit_model = PIT(exported_model, input_shape=input_shape).to(device)
    pit_model.train_features = True
    pit_model.train_rf = False
    pit_model.train_dilation = False

    size = torch.tensor(0, dtype=torch.float32)
    # size = torch.tensor(0)
    percentage = 30
    for layer in pit_model._target_layers:
        # size += layer.get_size()
        mem_in, mem_out, weights = layer.get_tot_mem()
        if (mem_in + mem_out + weights) < (44000+int(4400/100*percentage)):
            layer.TARGET = 44000
        elif (mem_in + mem_out + weights) < (400000+int(400000/100*percentage)):
            layer.TARGET = 400000
        else:
            layer.TARGET = 8000000
        print(f"Mem_in: {mem_in}, Mem_out: {mem_out}, W: {weights}, Target: {layer.TARGET}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS Search and Fine-Tuning')
    parser.add_argument('--epochs', type=int, default=100, help='Number of Training Epochs')
    parser.add_argument('--cd-size', type=float, default=2.4e-04, metavar='CD',
                        help='complexity decay size (default: 0.0)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Directory with Training Data')
    parser.add_argument('--size-target', type=float, default=0, metavar='ST',
                        help='target size (default: 0)')
    parser.add_argument('--latency-target', type=float, default=0, metavar='ST',
                        help='target latency (default: 0)')
    parser.add_argument('--model', type=str, default="const",
                        help='PIT, Supernet')
    args = parser.parse_args()
    main(args)
