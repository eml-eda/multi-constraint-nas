import argparse
import pathlib
from typing import Dict

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
import pytorch_benchmarks.anomaly_detection as amd
from pytorch_benchmarks.anomaly_detection.data import _file_to_vector_array
from pytorch_benchmarks.utils import AverageMeter, seed_all, accuracy, CheckPoint, EarlyStopping

from utils import evaluate, train_one_epoch
from hardware_model import get_latency_conv2D_GAP8, get_latency_Linear_GAP8, get_latency_conv2D_Diana, get_latency_Linear_Diana, get_latency_total
from hardware_model import compute_layer_latency_GAP8, compute_layer_latency_Diana, get_latency_binarized_supernet, get_size_binarized_supernet
from models import AutoEncoder
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np 
from sklearn import metrics
import math

def calculate_ae_accuracy(y_pred, y_true):
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, .01) * (np.amax(y_pred) - np.amin(y_pred))
    accuracy = 0
    for threshold in thresholds:
        y_pred_binary = (y_pred > threshold).astype(int)
        correct = np.sum(y_pred_binary == y_true)
        accuracy_tmp = 100 * correct / len(y_pred_binary)
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp
    return accuracy

def calculate_ae_pr_accuracy(y_pred, y_true):
    # initialize all arrays
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, .01) * (np.amax(y_pred) - np.amin(y_pred))
    accuracy = 0
    n_normal = np.sum(y_true == 0)
    precision = np.zeros(len(thresholds))
    recall = np.zeros(len(thresholds))

    # Loop on all the threshold values
    for threshold_item in range(len(thresholds)):
        threshold = thresholds[threshold_item]
        # Binarize the result
        y_pred_binary = (y_pred > threshold).astype(int)
        # Build matrix of TP, TN, FP and FN
        false_positive = np.sum((y_pred_binary[0:n_normal] == 1))
        true_positive = np.sum((y_pred_binary[n_normal:] == 1))
        false_negative = np.sum((y_pred_binary[n_normal:] == 0))
        # Calculate and store precision and recall
        precision[threshold_item] = true_positive / (true_positive + false_positive)
        recall[threshold_item] = true_positive / (true_positive + false_negative)
        # See if the accuracy has improved
        accuracy_tmp = 100 * (precision[threshold_item] + recall[threshold_item]) / 2
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp
    return accuracy

def calculate_ae_auc(y_pred, y_true):
    """
    Autoencoder ROC AUC calculation
    """
    # initialize all arrays
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.01, .01) * (np.amax(y_pred) - np.amin(y_pred))
    roc_auc = 0

    n_normal = np.sum(y_true == 0)
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))

    # Loop on all the threshold values
    for threshold_item in range(1, len(thresholds)):
        threshold = thresholds[threshold_item]
        # Binarize the result
        y_pred_binary = (y_pred > threshold).astype(int)
        # Build TP and FP
        tpr[threshold_item] = np.sum((y_pred_binary[n_normal:] == 1)
                                     ) / float(len(y_true) - n_normal)
        fpr[threshold_item] = np.sum((y_pred_binary[0:n_normal] == 1)) / float(n_normal)

    # Force boundary condition
    fpr[0] = 1
    tpr[0] = 1

    # Integrate
    for threshold_item in range(len(thresholds) - 1):
        roc_auc += .5 * (tpr[threshold_item] + tpr[threshold_item + 1]) * (
            fpr[threshold_item] - fpr[threshold_item + 1])
    return roc_auc

# Definition of evaluation function
def evaluate(
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        data,
        device: torch.device) -> Dict[str, float]:
    model.eval()
    test_metrics = {}
    for machine in data:
        y_pred = [0. for k in range(len(machine))]
        y_true = []
        machine_id = ''
        for file_idx, element in tqdm(enumerate(machine), total=len(machine), desc="preprocessing"):
            file_path, label, id = element
            machine_id = id[0]
            y_true.append(label[0].item())
            data_in = _file_to_vector_array(file_path[0],n_mels=128,frames=5,n_fft=1024,hop_length=512,power=2.0)
            data_in = data_in.astype('float32')
            data_in = torch.from_numpy(data_in)
            pred = model(data_in.to(device))
            data_in = data_in.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            errors = np.mean(np.square(data_in - pred), axis=1)
            y_pred[file_idx] = np.mean(errors)
        y_true = np.array(y_true, dtype='float64')
        y_pred = np.array(y_pred, dtype='float64')
        acc = calculate_ae_accuracy(y_pred, y_true)
        pr_acc = calculate_ae_pr_accuracy(y_pred, y_true)
        auc = calculate_ae_auc(y_pred, y_true)
        p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
        test_metrics[machine_id] = {
            'acc': acc,
            'pr_acc': pr_acc,
            'auc': auc,
            'p_auc': p_auc
        }
    performance = []
    for k, v in test_metrics.items():
        performance.append([v['auc'], v['p_auc'], v['acc'], v['pr_acc']])
    # calculate averages for AUCs and pAUCs
    averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
    test_metrics = {
        'acc': averaged_performance[2],
        'pr_acc': averaged_performance[3],
        'auc': averaged_performance[0],
        'p_auc': averaged_performance[1]
    }
    return test_metrics


# Definition of the function to train for one epoch
def train_one_epoch(
        epoch: int,
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dl: DataLoader,
        val_dl: DataLoader,
        test_dl: DataLoader,
        device: torch.device,
        args,
        increment_cd_size = None,
        increment_cd_ops = None) -> Dict[str, float]:
    model.train()
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0
    if args.l == "increasing":
        cd_size = min(args.cd_size/100 + increment_cd_size*epoch, args.cd_size)
        cd_ops = min(args.cd_ops/100 + increment_cd_ops*epoch, args.cd_ops)
    elif args.l == "const":
        cd_size = args.cd_size
        cd_ops = args.cd_ops
    # the goal is to arrive to the final cd_size and cd_ops in 1/5 of the total epochs
    # starting from 2 orders of magnitude less

    with tqdm(total=len(train_dl), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for audio in train_dl:
            step += 1
            tepoch.update(1)
            audio = audio.to(device)
            output = model(audio)
            loss_task = criterion(output, audio)
            
            if search:
                # Compute size-complexity loss
                if args.loss_type == "abs" and "mem_constraint" in args.loss_elements:
                    loss_reg = cd_size * torch.abs((model.get_size_binarized() - args.size_target))
                elif args.loss_type == "max" and "mem_constraint" in args.loss_elements:
                    loss_reg = cd_size * torch.max((model.get_size_binarized() - args.size_target), torch.FloatTensor([0]).to(device))[0]
                elif "mem_obj" in args.loss_elements:
                    loss_reg = cd_size * model.get_size_binarized()
                elif "mem" not in args.loss_elements:
                    loss_reg = 0

                # Compute latency-complexity loss
                if args.loss_type == "abs" and "lat_constraint" in args.loss_elements:
                    loss_ops = cd_ops * torch.abs((model.get_latency() - args.latency_target))
                elif args.loss_type == "max" and "lat_constraint" in args.loss_elements:
                    loss_ops = cd_ops * torch.max((model.get_latency() - args.latency_target), torch.FloatTensor([0]).to(device))[0]
                elif "lat_obj" in args.loss_elements:
                    loss_ops = cd_ops * model.get_latency()
                elif "lat" not in args.loss_elements:
                    loss_ops = 0

                if args.model == "PIT" or args.gumbel == "True":
                    loss_icv = 0
                elif args.model == "Supernet":
                    loss_icv = 0.5 * model.get_total_icv()

                loss = loss_task + loss_ops + loss_reg + loss_icv
            else:

                loss = loss_task
                loss_reg = 0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avgloss.update(loss, audio.size(0))
            avglosstask.update(loss_task, audio.size(0))
            avglossreg.update(loss_reg, audio.size(0))
            if step % 100 == 99:
                tepoch.set_postfix({'loss': avgloss,
                                    'loss_task': avglosstask,
                                    'loss_reg': avglossreg})
        val_metrics = evaluate(search, model, criterion, test_dl, device)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        final_metrics = {
            'loss': avgloss.get(),
            'loss_task': avglosstask.get(),
            'loss_reg': avglossreg.get()
        }
        final_metrics.update(val_metrics)
        tepoch.set_postfix(final_metrics)
        tepoch.close()
        print(final_metrics)
        return final_metrics

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
    datasets = amd.get_data(data_dir=data_dir)
    dataloaders = amd.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders

    # Get the Model
    if args.model == "PIT":
        model = amd.get_reference_model('autoencoder')
        model = model.to(device)
        # Model Summary
        input_example = torch.unsqueeze(datasets[0][0], 0).to(device)
        input_shape = datasets[0][0].numpy().shape
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
            model = AutoEncoder(gumbel = True)
        else:
            model = AutoEncoder(gumbel = False)
        model = model.to(device)

        PITSuperNet.get_macs_binarized = PITSuperNet.get_macs
        PITSuperNet.get_latency = PITSuperNet.get_macs
        PITSuperNet.get_size_binarized = PITSuperNet.get_size

        # Model Summary
        input_example = torch.unsqueeze(datasets[0][0], 0).to(device)
        input_shape = datasets[0][0].numpy().shape

    print(summary(model, input_example, show_input=False, show_hierarchical=True))

    # Warmup Loop
    criterion = amd.get_default_criterion()
    optimizer = amd.get_default_optimizer(model)
    name = f"ck_amd_{args.model}"
    warmup_checkpoint = CheckPoint('./warmup_checkpoints', model, optimizer, 'max',fmt=name+'_{epoch:03d}.pt')
    skip_warmup = True
    if pathlib.Path(f'./warmup_checkpoints/final_best_warmup_amd_{args.model}.ckp').exists():
        warmup_checkpoint.load(f'./warmup_checkpoints/final_best_warmup_amd_{args.model}.ckp')
        print("Skipping warmup")
    else:
        skip_warmup = False
        print("Running warmup")

    if not skip_warmup:
        for epoch in range(N_EPOCHS):
            metrics = train_one_epoch(
                epoch, False, model, criterion, optimizer, train_dl, val_dl, test_dl, device, args, 1, 1)
            warmup_checkpoint(epoch, metrics['val_acc'])
        warmup_checkpoint.load_best()
        warmup_checkpoint.save(f'./warmup_checkpoints/final_best_warmup_amd_{args.model}.ckp')

    test_metrics = evaluate(False, model, criterion, test_dl, device)
    print("Warmup Test Set AUC:", test_metrics['auc'])
    print("Warmup Test Set Accuracy:", test_metrics['acc'])
    print("Warmup Test Set P AUC:", test_metrics['p_auc'])
    print("Warmup Test Set P Accuracy:", test_metrics['pr_acc'])

    # Convert the model to PIT
    if args.model == "PIT":
        pit_model = PIT(model, input_shape=input_shape)
        pit_model = pit_model.to(device)
        pit_model.train_features = True
        pit_model.train_rf = False
        pit_model.train_dilation = False
    elif args.model == "Supernet":
        pit_model = PITSuperNet(model, input_shape=input_shape, autoconvert_layers = False)
        pit_model = pit_model.to(device)
    print(summary(pit_model, input_example, show_input=False, show_hierarchical=True))

    # Search Loop
    criterion = amd.get_default_criterion()
    param_dicts = [
        {'params': pit_model.nas_parameters(), 'weight_decay': 0},
        {'params': pit_model.net_parameters()}]
    optimizer = torch.optim.Adam(param_dicts)
    # Set EarlyStop with a patience of 20 epochs and CheckPoint
    earlystop = EarlyStopping(patience=20, mode='max')
    name = f"ck_amd_opt_{args.model}_{args.loss_type}_targets_{args.loss_elements}_{args.l}_size_{args.size_target}_lat_{args.latency_target}"
    search_checkpoint = CheckPoint('./search_checkpoints', pit_model, optimizer, 'max', fmt=name+'_{epoch:03d}.pt')
    print("Initial model size:", pit_model.get_size_binarized())
    print("Initial model MACs:", pit_model.get_macs_binarized())
    print("Initial model latency:", pit_model.get_latency())
    print("Initial model MACs/cycle:", pit_model.get_macs_binarized()/pit_model.get_latency())
    increment_cd_size = (args.cd_size*99/100)/int(N_EPOCHS/2)
    increment_cd_ops = (args.cd_ops*99/100)/int(N_EPOCHS/2)
    temp = 1
    for epoch in range(N_EPOCHS):
        metrics = train_one_epoch(
            epoch, True, pit_model, criterion, optimizer, train_dl, val_dl, test_dl, device, args, increment_cd_size, increment_cd_ops)
        if args.model == "Supernet":
            temp = temp * math.exp(-0.05)
            pit_model.update_softmax_temperature(temp)
            for module in pit_model.modules(): 
                if isinstance(module, PITSuperNetCombiner):
                    print(nn.functional.softmax(module.alpha/module.softmax_temperature, dim=0))
                    print(module.softmax_temperature)
        if epoch > int(N_EPOCHS/2+N_EPOCHS/4):
            search_checkpoint(epoch, metrics['val_acc'])
            if earlystop(metrics['val_acc']):
                break

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
    print("Search Test Set AUC:", test_metrics['auc'])
    print("Search Test Set Accuracy:", test_metrics['acc'])
    print("Search Test Set P AUC:", test_metrics['p_auc'])
    print("Search Test Set P Accuracy:", test_metrics['pr_acc'])

    # Convert pit model into pytorch model
    exported_model = pit_model.arch_export()
    exported_model = exported_model.to(device)
    print(summary(exported_model, input_example, show_input=False, show_hierarchical=True))

    # Fine-tuning
    criterion = amd.get_default_criterion()
    optimizer = amd.get_default_optimizer(exported_model)
    name = f"ck_amd_opt_{args.model}_{args.loss_type}_targets_{args.loss_elements}_{args.l}_size_{args.size_target}_lat_{args.latency_target}"
    finetune_checkpoint = CheckPoint('./finetuning_checkpoints', exported_model, optimizer, 'max', fmt=name+'_{epoch:03d}.pt')
    earlystop = EarlyStopping(patience=20, mode='max')
    for epoch in range(N_EPOCHS):
        metrics = train_one_epoch(
            epoch, False, exported_model, criterion, optimizer, train_dl, val_dl, test_dl, device, args, increment_cd_size, increment_cd_ops)
        print("epoch:", epoch)
        if epoch > 0:
            finetune_checkpoint(epoch, metrics['val_acc'])
            if earlystop(metrics['val_acc']):
                break
    finetune_checkpoint.load_best()
    name = f"best_final_ck_amd_opt_{args.model}_{args.loss_type}_targets_{args.loss_elements}_{args.l}_size_{args.size_target}_lat_{args.latency_target}.ckp"
    finetune_checkpoint.save('./finetuning_checkpoints/'+name)
    test_metrics = evaluate(False, exported_model, criterion, test_dl, device)
    print("Fine-tuning  Test Set AUC:", test_metrics['auc'])
    print("Fine-tuning  Test Set Accuracy:", test_metrics['acc'])
    print("Fine-tuning  Test Set P AUC:", test_metrics['p_auc'])
    print("Fine-tuning  Test Set P Accuracy:", test_metrics['pr_acc'])
    print("Fine-tuning PLiNIO size:", pit_model.get_size_binarized())
    print("Fine-tuning PLiNIO MACs:", pit_model.get_macs_binarized())
    print("Fine-tuning PLiNIO latency:", pit_model.get_latency())
    print("Fine-tuning PLiNIO MACs/cycle:", pit_model.get_macs_binarized()/pit_model.get_latency())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS Search and Fine-Tuning')
    parser.add_argument('--epochs', type=int, help='Number of Training Epochs')
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
    parser.add_argument('--model', type=str, default="const",
                        help='PIT, Supernet')
    parser.add_argument('--hardware', type=str, default="const",
                        help='GAP8, Diana, None')
    parser.add_argument('--gumbel', type=str, default="False",
                        help='True or False')
    args = parser.parse_args()
    main(args)
