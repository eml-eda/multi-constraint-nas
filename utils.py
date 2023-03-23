import argparse
import pathlib
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_benchmarks.utils import AverageMeter,accuracy

# Definition of evaluation function
def evaluate(
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device) -> Dict[str, float]:
    model.eval()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for audio, target in data:
            step += 1
            audio, target = audio.to(device), target.to(device)
            output = model(audio)
            loss_task = criterion(output, target)
            loss = loss_task
            loss_reg = 0.
            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], audio.size(0))
            avgloss.update(loss, audio.size(0))
            avglosstask.update(loss_task, audio.size(0))
            avglossreg.update(loss_reg, audio.size(0))
        final_metrics = {
            'loss': avgloss.get(),
            'loss_task': avglosstask.get(),
            'loss_reg': avglossreg.get(),
            'acc': avgacc.get(),
        }
    return final_metrics


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
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    avglossops = AverageMeter('2.5f')
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
        for audio, target in train_dl:
            step += 1
            tepoch.update(1)
            audio, target = audio.to(device), target.to(device)
            output = model(audio)
            loss_task = criterion(output, target)
            
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
                    import pdb;pdb.set_trace()
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
                loss_ops = 0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], audio.size(0))
            avgloss.update(loss, audio.size(0))
            avglosstask.update(loss_task, audio.size(0))
            avglossreg.update(loss_reg, audio.size(0))
            avglossops.update(loss_ops, audio.size(0))
            if step % 100 == 99:
                tepoch.set_postfix({'loss': avgloss,
                                    'loss_task': avglosstask,
                                    'loss_reg': avglossreg,
                                    'loss_ops': avglossops,
                                    'acc': avgacc})
        val_metrics = evaluate(search, model, criterion, val_dl, device)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        test_metrics = evaluate(search, model, criterion, test_dl, device)
        test_metrics = {'test_' + k: v for k, v in test_metrics.items()}
        final_metrics = {
            'loss': avgloss.get(),
            'loss_task': avglosstask.get(),
            'loss_reg': avglossreg.get(),
            'loss_ops': avglossops.get(),
            'acc': avgacc.get(),
        }
        final_metrics.update(val_metrics)
        tepoch.set_postfix(final_metrics)
        tepoch.close()
        print(final_metrics)
        print(test_metrics)
        return final_metrics
