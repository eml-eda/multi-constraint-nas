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

from flexnas.methods import PIT
from flexnas.methods import PITSuperNet
from flexnas.methods.pit.nn import PITConv2d, PITLinear
from flexnas.methods.pit_supernet.nn import PITSuperNetCombiner
import pytorch_benchmarks.tiny_imagenet as icl
from pytorch_benchmarks.utils import AverageMeter, seed_all, accuracy, CheckPoint, EarlyStopping
from hardware_model import get_memory_layer_constraints, get_Lx_level_constraint_conv, get_Lx_level_constraint_linear, get_tot_mem_conv, get_tot_mem_linear

from utils import evaluate, train_one_epoch_finetuning
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
    datasets2 = icl.get_data(data_dir=data_dir)
    dataloaders2 = icl.build_dataloaders(datasets2)
    train_dl, val_dl, test_dl_all = dataloaders2
    def load(model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    model = icl.get_reference_model('resnet_18')
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
        if f"ck_icl_opt_{args.model}_max_targets_mem_constraint+lat_constraint_increasing_size_{args.size_target}_lat_{args.latency_target}_" in file:
            if int(file.split(".")[-2].split("_")[-1]) > log:
                log = int(file.split(".")[-2].split("_")[-1])
    if log < 100:
        pit_model=load(pit_model, f'search_checkpoints/ck_icl_opt_{args.model}_max_targets_mem_constraint+lat_constraint_increasing_size_{args.size_target}_lat_{args.latency_target}_0{log}.pt')
    else:
        pit_model=load(pit_model, f'search_checkpoints/ck_icl_opt_{args.model}_max_targets_mem_constraint+lat_constraint_increasing_size_{args.size_target}_lat_{args.latency_target}_{log}.pt')
    
    print("final model size:", pit_model.get_size_binarized())
    print("final model MACs:", pit_model.get_macs_binarized())
    # Convert pit model into pytorch model
    exported_model = pit_model.arch_export()
    exported_model = exported_model.to(device)
    print(summary(exported_model, input_example, show_input=False, show_hierarchical=True))
    log = 0
    for file in os.listdir("finetuning_checkpoints"):
        if f"ck_icl_opt_{args.model}_max_targets_mem_constraint+lat_constraint_increasing_size_{args.size_target}_lat_{args.latency_target}_" in file:
            if int(file.split(".")[-2].split("_")[-1]) > log:
                log = int(file.split(".")[-2].split("_")[-1])
    if log < 100:
        exported_model=load(exported_model, f'finetuning_checkpoints/ck_icl_opt_{args.model}_max_targets_mem_constraint+lat_constraint_increasing_size_{args.size_target}_lat_{args.latency_target}_0{log}.pt')
    else:
        exported_model=load(exported_model, f'finetuning_checkpoints/ck_icl_opt_{args.model}_max_targets_mem_constraint+lat_constraint_increasing_size_{args.size_target}_lat_{args.latency_target}_{log}.pt')
    test_metrics2 = evaluate(True, exported_model, criterion, test_dl_all, device)
    print("Search Test Set Accuracy:", test_metrics2['acc'])

    PITConv2d.get_tot_mem = get_tot_mem_conv
    PITLinear.get_tot_mem = get_tot_mem_linear
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
        size = layer.get_tot_mem()
        if size < (44000+int(4400/100*percentage)):
            layer.TARGET = 44000
        elif size < (400000+int(400000/100*percentage)):
            layer.TARGET = 400000
        else:
            layer.TARGET = size
        print(f"Size: {size}, Target: {layer.TARGET}")
    return 0
    # Search Loop
    criterion = icl.get_default_criterion()
    param_dicts = [
        {'params': pit_model.nas_parameters(), 'weight_decay': 0},
        {'params': pit_model.net_parameters()}]
    optimizer = torch.optim.Adam(param_dicts, lr=0.001, weight_decay=1e-4)
    scheduler = icl.get_default_scheduler(optimizer)
    # Set EarlyStop with a patience of 20 epochs and CheckPoint
    earlystop = EarlyStopping(patience=20, mode='max')
    name = f"ck_icl_opt_max_targets_size_{args.size_target}_lat_{args.latency_target}"
    search_checkpoint = CheckPoint('./search_GAP8finetuning_checkpoints', pit_model, optimizer, 'max', fmt=name+'_{epoch:03d}.pt')
    print("Initial model size:", pit_model.get_size_binarized())
    increment_cd_size = (args.cd_size*99/100)/int(N_EPOCHS/2)

    for epoch in range(N_EPOCHS):
        metrics = train_one_epoch_finetuning(epoch, True, pit_model, criterion, optimizer, train_dl, val_dl, test_dl_all, device, args, increment_cd_size)
        if epoch > int(N_EPOCHS/2+N_EPOCHS/4):
            search_checkpoint(epoch, metrics['val_acc'])
            if earlystop(metrics['val_acc']):
                break
        scheduler.step()
        print("epoch:", epoch)
        print("model size:", pit_model.get_size_binarized())
        print(f"cd_size:  {min(args.cd_size/100 + increment_cd_size*epoch, args.cd_size)}")
    print("Load best model")
    search_checkpoint.load_best()
    print("final model size:", pit_model.get_size_binarized())
    print("final architectural summary:")
    print(pit_model)
    test_metrics2 = evaluate(True, pit_model, criterion, test_dl_all, device)
    print("Search Test Set Accuracy:", test_metrics2['acc'])

    # Convert pit model into pytorch model
    exported_model = pit_model.arch_export()
    exported_model = exported_model.to(device)
    print(summary(exported_model, input_example, show_input=False, show_hierarchical=True))

    # Fine-tuning
    criterion = icl.get_default_criterion()
    optimizer = icl.get_default_optimizer(exported_model)
    scheduler = icl.get_default_scheduler(optimizer)
    name = f"ck_icl_opt_max_targets_size_{args.size_target}_lat_{args.latency_target}"
    finetune_checkpoint = CheckPoint('./finetuning_GAP8finetuning_checkpoints', exported_model, optimizer, 'max', fmt=name+'_{epoch:03d}.pt')
    earlystop = EarlyStopping(patience=20, mode='max')
    for epoch in range(N_EPOCHS):
        metrics = train_one_epoch_finetuning(epoch, False, exported_model, criterion, optimizer, train_dl, val_dl, test_dl_all, device, args, increment_cd_size, increment_cd_ops)
        scheduler.step()
        print("epoch:", epoch)
        if epoch > 0:
            finetune_checkpoint(epoch, metrics['val_acc'])
            if earlystop(metrics['val_acc']):
                break
    finetune_checkpoint.load_best()
    name = f"best_final_ck_icl_opt_max_targets_size_{args.size_target}_lat_{args.latency_target}.ckp"
    finetune_checkpoint.save('./finetuning_checkpoints/'+name)
    test_metrics2 = evaluate(False, exported_model, criterion, test_dl_all, device)
    print("Fine-tuning Test Set Accuracy:", test_metrics2['acc'])
    print("Fine-tuning PLiNIO size:", pit_model.get_size_binarized())
    print("Fine-tuning PLiNIO MACs:", pit_model.get_macs_binarized())


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
