import argparse
import pathlib
from typing import Dict

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

from utils.utils import evaluate, train_one_epoch
from utils.hardware_model import get_latency_conv2D_GAP8, get_latency_Linear_GAP8, get_latency_conv2D_Diana, get_latency_Linear_Diana, get_latency_total
from utils.hardware_model import compute_layer_latency_GAP8, compute_layer_latency_Diana, get_latency_binarized_supernet, get_size_binarized_supernet
from utils.models import ResNet8PITSN

def main(args):
    DATA_DIR = args.data_dir
    N_EPOCHS = args.epochs

    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    # Ensure determinstic execution
    seed_all(seed=14)

    # Get the Data
    data_dir = DATA_DIR
    datasets = icl.get_data(inp_res=224)
    dataloaders = icl.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders


    # Get the Model
    if args.model == "PIT":
        config = {'pretrained': True, 'std_head': True}
        model = icl.get_reference_model('resnet_18', model_config=config)
        model = model.to(device)
        # Model Summary
        input_example = torch.unsqueeze(datasets[0][0][0], 0).to(device)
        input_shape = datasets[0][0][0].numpy().shape
        # Convert the model to PIT
        if args.hardware == "GAP8":
            PITConv2d.get_latency = get_latency_conv2D_GAP8
            PITLinear.get_latency = get_latency_Linear_GAP8
        elif args.hardware == "Diana":
            PITConv2d.get_latency = get_latency_conv2D_Diana
            PITLinear.get_latency = get_latency_Linear_Diana
        elif args.hardware == "None":
            PITConv2d.get_latency = PITConv2d.get_macs_binarized
            PITLinear.get_latency = PITLinear.get_macs_binarized
        PIT.get_latency = get_latency_total
    elif args.model == "Supernet":
        if args.hardware == "GAP8":
            PITSuperNetCombiner.compute_layers_macs = compute_layer_latency_GAP8
        elif args.hardware == "Diana":
            PITSuperNetCombiner.compute_layers_macs = compute_layer_latency_Diana
        elif args.hardware == "None":
            pass
        PITSuperNetCombiner.get_size = get_size_binarized_supernet
        PITSuperNetCombiner.get_macs = get_latency_binarized_supernet
        if args.gumbel == "True":
            model = ResNet8PITSN(gumbel = True)
        else:
            model = ResNet8PITSN(gumbel = False)

        model = model.to(device)
        PITSuperNet.get_macs_binarized = PITSuperNet.get_macs
        PITSuperNet.get_latency = PITSuperNet.get_macs
        PITSuperNet.get_size_binarized = PITSuperNet.get_size

        # Model Summary
        input_example = torch.unsqueeze(datasets[0][0][0], 0).to(device)
        input_shape = datasets[0][0][0].numpy().shape

    # print(summary(model, input_example, show_input=False, show_hierarchical=True))

    # Warmup Loop
    criterion = icl.get_default_criterion()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    name = f"ck_tiny_224_{args.model}"
    warmup_checkpoint = CheckPoint('./warmup_checkpoints', model, optimizer, 'max',fmt=name+'_{epoch:03d}.pt')
    skip_warmup = True
    if pathlib.Path(f'./warmup_checkpoints/final_best_warmup_tiny_{args.model}.ckp').exists():
        #warmup_checkpoint.load(f'./warmup_checkpoints/final_best_warmup_tiny_{args.model}.ckp')
        print("Skipping warmup 224")
    else:
        skip_warmup = False
        print("Running warmup 224")

    if not skip_warmup:
        for epoch in range(N_EPOCHS):
            metrics = train_one_epoch(
                epoch, False, model, criterion, optimizer, train_dl, val_dl, test_dl, device, args, 1, 1)
            scheduler.step()
            warmup_checkpoint(epoch, metrics['val_acc'])
        warmup_checkpoint.load_best()

    test_metrics = evaluate(False, model, criterion, test_dl, device)
    
    print("Warmup Test Set 224 Loss:", test_metrics['loss'])
    print("Warmup Test Set 224 Accuracy:", test_metrics['acc'])
    # Get the Data with 64pixel resolution as full-imagenet
    datasets = icl.get_data(inp_res=64)
    dataloaders = icl.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders

    # Now, get Model with reduced ResNet18 head
    config = {'pretrained': True,
            'state_dict': model.state_dict(),  # state-dict of previously trained model
            'std_head': False}
    model = icl.get_reference_model('resnet_18', model_config=config)
    model = model.to(device)


    # Model Summary
    input_example = torch.unsqueeze(datasets[0][0][0], 0).to(device)
    input_shape = datasets[0][0][0].numpy().shape

    # Get Training Settings
    criterion = icl.get_default_criterion()
    optimizer = icl.get_default_optimizer(model)
    scheduler = icl.get_default_scheduler(optimizer)

    name = f"ck_tiny_{args.model}"
    warmup_checkpoint = CheckPoint('./warmup_checkpoints', model, optimizer, 'max',fmt=name+'_{epoch:03d}.pt')
    if pathlib.Path(f'./warmup_checkpoints/final_best_warmup_tiny_{args.model}.ckp').exists():
        warmup_checkpoint.load(f'./warmup_checkpoints/final_best_warmup_tiny_{args.model}.ckp')
        print("Skipping warmup 64")
    else:
        skip_warmup = False
        print("Running warmup 64")
    if not skip_warmup:
        for epoch in range(N_EPOCHS):
            metrics = train_one_epoch(
                epoch, False, model, criterion, optimizer, train_dl, val_dl, test_dl, device, args, 1, 1)
            scheduler.step()
            warmup_checkpoint(epoch, metrics['val_acc'])
        warmup_checkpoint.load_best()
        warmup_checkpoint.save(f'./warmup_checkpoints/final_best_warmup_tiny_{args.model}.ckp')

    test_metrics = evaluate(False, model, criterion, test_dl, device)
    
    print("Warmup Test Set Loss:", test_metrics['loss'])
    print("Warmup Test Set Accuracy:", test_metrics['acc'])


    # Convert the model to PIT
    if args.model == "PIT":
        pit_model = PIT(model, input_shape=input_shape)
        pit_model = pit_model.to(device)
        pit_model.train_features = True
        pit_model.train_rf = False
        pit_model.train_dilation = False
    print(summary(pit_model, input_example, show_input=False, show_hierarchical=True))

    # Search Loop
    criterion = icl.get_default_criterion()
    param_dicts = [
        {'params': pit_model.nas_parameters(), 'weight_decay': 0},
        {'params': pit_model.net_parameters()}]
    
    optimizer = torch.optim.SGD(param_dicts, lr=0.001, momentum=0.9, weight_decay=1e-4)
    arch_optimizer = torch.optim.Adam(pit_model.nas_parameters(), lr=0.001)
    scheduler = icl.get_default_scheduler(optimizer)
    # Set EarlyStop with a patience of 20 epochs and CheckPoint
    earlystop = EarlyStopping(patience=10, mode='max')
    name = f"ck_tiny_opt_{args.model}_{args.loss_type}_targets_{args.loss_elements}_{args.l}_size_{args.size_target}_lat_{args.latency_target}"
    search_checkpoint = CheckPoint('./search_checkpoints', pit_model, optimizer, 'max', fmt=name+'_{epoch:03d}.pt')
    print("Initial model size:", pit_model.get_size_binarized())
    print("Initial model MACs:", pit_model.get_macs_binarized())
    print("Initial model latency:", pit_model.get_latency())
    print("Initial model MACs/cycle:", pit_model.get_macs_binarized()/pit_model.get_latency())
    # test_metrics = evaluate(False, pit_model, criterion, test_dl, device)
    
    # print("Initial Test Set Loss:", test_metrics['loss'])
    # print("Initial Test Set Accuracy:", test_metrics['acc'])
    increment_cd_size = (args.cd_size*99/100)/int(N_EPOCHS/2)
    increment_cd_ops = (args.cd_ops*99/100)/int(N_EPOCHS/2)
    # search_checkpoint.epoch = 26
    # search_checkpoint.update_best_path(26, 1)
    temp = 1
    for epoch in range(N_EPOCHS):
        metrics = train_one_epoch(
            epoch, True, pit_model, criterion, optimizer, train_dl, val_dl, test_dl, device, args, increment_cd_size, increment_cd_ops, arch_optimizer)
            
        if epoch > int(N_EPOCHS/2):
            search_checkpoint(epoch, metrics['val_acc'])
            if earlystop(metrics['val_acc']):
                break

        scheduler.step()
        # print("architectural summary:")
        # print(pit_model)
        print("epoch:", epoch)
        print("model size:", pit_model.get_size_binarized())
        print("model MACs:", pit_model.get_macs_binarized())
        print("model latency:", pit_model.get_latency())
        print("model MACs/cycle:", pit_model.get_macs_binarized()/pit_model.get_latency())
        print(f"cd_size:  {min(args.cd_size/100 + increment_cd_size*epoch, args.cd_size)} cd_ops: {min(args.cd_ops/100 + increment_cd_ops*epoch, args.cd_ops)}")
    print("Load best model")
    search_checkpoint.load_best()
    print("final model size:", pit_model.get_size_binarized())
    print("final model MACs:", pit_model.get_macs_binarized())
    print("final model latency:", pit_model.get_latency())
    print("final model MACs/cycle:", pit_model.get_macs_binarized()/pit_model.get_latency())
    print("final architectural summary:")
    print(pit_model)
    test_metrics = evaluate(True, pit_model, criterion, test_dl, device)
    
    print("Search Test Set Loss:", test_metrics['loss'])
    print("Search Test Set Accuracy:", test_metrics['acc'])

    # Convert pit model into pytorch model
    exported_model = pit_model.arch_export()
    exported_model = exported_model.to(device)
    print(summary(exported_model, input_example, show_input=False, show_hierarchical=True))

    # Fine-tuning
    criterion = icl.get_default_criterion()
    optimizer = icl.get_default_optimizer(exported_model)
    scheduler = icl.get_default_scheduler(optimizer)
    name = f"ck_tiny_opt_{args.model}_{args.loss_type}_targets_{args.loss_elements}_{args.l}_size_{args.size_target}_lat_{args.latency_target}"
    finetune_checkpoint = CheckPoint('./finetuning_checkpoints', exported_model, optimizer, 'max', fmt=name+'_{epoch:03d}.pt')
    earlystop = EarlyStopping(patience=20, mode='max')
    for epoch in range(N_EPOCHS):
        metrics = train_one_epoch(
            epoch, False, exported_model, criterion, optimizer, train_dl, val_dl, test_dl, device, args, increment_cd_size, increment_cd_ops)
        scheduler.step()
        print("epoch:", epoch)
        if epoch > 0:
            finetune_checkpoint(epoch, metrics['val_acc'])
            if earlystop(metrics['val_acc']):
                break
    finetune_checkpoint.load_best()
    name = f"best_final_ck_tiny_opt_{args.model}_{args.loss_type}_targets_{args.loss_elements}_{args.l}_size_{args.size_target}_lat_{args.latency_target}.ckp"
    finetune_checkpoint.save('./finetuning_checkpoints/'+name)
    test_metrics = evaluate(False, exported_model, criterion, test_dl, device)
    print("Fine-tuning Test Set Loss:", test_metrics['loss'])
    print("Fine-tuning Test Set Accuracy:", test_metrics['acc'])
    print("Fine-tuning PLiNIO size:", pit_model.get_size_binarized())
    print("Fine-tuning PLiNIO MACs:", pit_model.get_macs_binarized())
    print("Fine-tuning PLiNIO latency:", pit_model.get_latency())
    print("Fine-tuning PLiNIO MACs/cycle:", pit_model.get_macs_binarized()/pit_model.get_latency())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS Search and Fine-Tuning')
    parser.add_argument('--epochs', type=int, default = 50, help='Number of Training Epochs')
    parser.add_argument('--cd-size', type=float, default=0.0, metavar='CD',
                        help='complexity decay size (default: 0.0)')
    parser.add_argument('--cd-ops', type=float, default=0.0, metavar='CD',
                        help='complexity decay ops (default: 0.0)')
    parser.add_argument('--size-target', type=float, default=0, metavar='ST',
                        help='target size (default: 0)')
    parser.add_argument('--latency-target', type=float, default=0, metavar='ST',
                        help='target latency (default: 0)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Directory with Training Data')
    parser.add_argument('--loss_type', type=str, default="max",
                        help='abs, max')
    parser.add_argument('--loss_elements', type=str, default="mem",
                        help='loss type: mem_constraint, mem_obj, lat_constraint, lat_obj,and fusion')
    parser.add_argument('--l', type=str, default="const",
                        help='const, increasing')
    parser.add_argument('--model', type=str, default="PIT",
                        help='PIT, Supernet')
    parser.add_argument('--hardware', type=str, default="None",
                        help='GAP8, Diana, None')
    parser.add_argument('--gumbel', type=str, default="False",
                        help='True or False')
    args = parser.parse_args()
    main(args)
