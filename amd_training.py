import argparse
import pathlib
from typing import Dict

from pytorch_model_summary import summary
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.optim as optim

from flexnas.methods import PIT
import pytorch_benchmarks.anomaly_detection as amd
from pytorch_benchmarks.utils import AverageMeter, seed_all, CheckPoint, EarlyStopping


def evaluate(
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device) -> Dict[str, float]:
    model.eval()
    avgloss = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for audio in data:
            step += 1
            audio = audio.to(device)
            output, loss_task = _run_model(model, audio, audio, criterion, device)
            loss = loss_task
            loss_reg = 0.
            avgloss.update(loss, audio.size(0))
        final_metrics = {
            'loss': avgloss.get()
        }
    return final_metrics


def _run_model(model, audio, target, criterion, device):
    output = model(audio)
    loss = criterion(output, target)
    return output, loss


def train_one_epoch(
        epoch: int,
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train: DataLoader,
        val: DataLoader,
        device: torch.device,
        args) -> Dict[str, float]:
    model.train()
    avgloss = AverageMeter('2.5f')
    step = 0
    with tqdm(total=len(train), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for audio in train:
            step += 1
            tepoch.update(1)
            audio = audio.to(device)
            output, loss_task = _run_model(model, audio, audio, criterion, device)
            if search:
                # Compute size-complexity loss with constraint
                loss_reg = args.cd_size * torch.abs((model.get_size() - args.size_target))
                # Compute ops-complexity loss with constraint
                loss_ops = args.cd_ops * model.get_macs()
                loss = loss_task + loss_ops + loss_reg
            else:
                loss = loss_task
                loss_reg = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avgloss.update(loss, audio.size(0))
            if step % 100 == 99:
                tepoch.set_postfix({'loss': avgloss})
        val_metrics = amd.evaluate(model, criterion, val, device)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        final_metrics = {
            'loss': avgloss.get(),
        }
        final_metrics.update(val_metrics)
        tepoch.set_postfix(final_metrics)
        tepoch.close()
    return final_metrics


def test_results(test_metrics):
    print("test_metrics: ", test_metrics)
    performance = []
    print("\nTest results:")
    for k, v in test_metrics.items():
        print('machine id={}, accuracy={:.5f}, precision/accuracy={:.5f}, auc={:.5f}, p_auc={:.5f}'
              .format(k, v['acc'], v['pr_acc'], v['auc'], v['p_auc']))
        performance.append([v['auc'], v['p_auc']])
    # calculate averages for AUCs and pAUCs
    averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
    auc = averaged_performance[0]
    pAuc = averaged_performance[1]
    return auc, pAuc


def main(args):
    DATA_DIR = args.data_dir
    N_EPOCHS = args.epochs

    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    # Ensure determinstic execution
    seed_all(seed=42)

    # Get the Data
    data_dir = DATA_DIR
    datasets = amd.get_data(data_dir=data_dir)
    dataloaders = amd.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders

    # Get the Model
    model = amd.get_reference_model('autoencoder')
    model = model.to(device)

    # Model Summary
    input_example = torch.unsqueeze(datasets[0][0], 0).to(device)
    input_shape = datasets[0][0].numpy().shape
    print(summary(model, input_example, show_input=False, show_hierarchical=True))

    # Warmup Loop
    criterion = amd.get_default_criterion()
    optimizer = amd.get_default_optimizer(model)

    warmup_checkpoint = CheckPoint(f'./warmup_checkpoints', model, optimizer, 'min', fmt='ck_amd_{epoch:03d}.pt')
    skip_warmup = True
    if pathlib.Path(f'.warmup_checkpoints/final_best_warmup_amd.ckp').exists():
        warmup_checkpoint.load(f'.warmup_checkpoints/final_best_warmup_amd.ckp')
        print("Skipping warmup")
    else:
        skip_warmup = False
        print("Running warmup")

    if not skip_warmup:
        for epoch in range(N_EPOCHS):
            train_metrics = train_one_epoch(
                epoch, False, model, criterion, optimizer, train_dl, val_dl, device, args)
            warmup_checkpoint(epoch, train_metrics['val_loss'])
        warmup_checkpoint.load_best()
        warmup_checkpoint.save(f'.warmup_checkpoints/final_best_warmup.ckp')

    test_metrics = amd.test(test_dl, model)
    auc, pAuc = test_results(test_metrics)
    print("Warmup Test Set Average AUC: ", auc)
    print("Warmup Test Set Average pAUC: ", pAuc)

    # Convert the model to PIT
    pit_model = PIT(model, input_shape=input_shape)
    pit_model = pit_model.to(device)
    pit_model.train_features = True
    pit_model.train_rf = False
    pit_model.train_dilation = False
    print(summary(pit_model, input_example, show_input=False, show_hierarchical=True))

    # Search Loop
    criterion = amd.get_default_criterion()
    param_dicts = [
        {'params': pit_model.nas_parameters(), 'weight_decay': 0},
        {'params': pit_model.net_parameters()}]
    optimizer = torch.optim.Adam(param_dicts)
    # Set EarlyStop with a patience of 20 epochs and CheckPoint
    earlystop = EarlyStopping(patience=20, mode='max')
    search_checkpoint = CheckPoint(f'./search_checkpoints', pit_model, optimizer, 'min', fmt='ck_amd_{epoch:03d}.pt')
    for epoch in range(N_EPOCHS):
        train_metrics = train_one_epoch(
            epoch, True, pit_model, criterion, optimizer, train_dl, val_dl, device,args)

        if epoch > 5:
            search_checkpoint(epoch, train_metrics['val_loss'])
            if earlystop(train_metrics['val_loss']):
                break

        print("architectural summary:")
        print(pit_model)
        print("model size:", pit_model.get_size())
    print("Load best model")
    search_checkpoint.load_best()
    print("final architectural summary:")
    print(pit_model)

    test_metrics = amd.test(test_dl, pit_model)
    auc, pAuc = test_results(test_metrics)
    print("Search Test Set Average AUC: ", auc)
    print("Search Test Set Average pAUC: ", pAuc)

    # Convert pit model into pytorch model
    exported_model = pit_model.arch_export()
    exported_model = exported_model.to(device)
    print(summary(exported_model, input_example, show_input=False, show_hierarchical=True))

    # Fine-tuning
    criterion = amd.get_default_criterion()
    optimizer = amd.get_default_optimizer(exported_model)
    finetune_checkpoint = CheckPoint('./finetuning_checkpoints', pit_model, optimizer, 'min', fmt='ck_amd_{epoch:03d}.pt')
    earlystop = EarlyStopping(patience=50, mode='max')
    for epoch in range(N_EPOCHS):
        metrics = train_one_epoch(
            epoch, False, exported_model, criterion, optimizer, train_dl, val_dl, device, args)
        if epoch > 0:
            finetune_checkpoint(epoch, metrics['val_acc'])
            if earlystop(metrics['val_acc']):
                break
    test_metrics = amd.test(test_dl, exported_model)
    auc, pAuc = test_results(test_metrics)
    print("Fine-tuning Test Set Average AUC: ", auc)
    print("Fine-tuning Test Set Average pAUC: ", pAuc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS Search and Fine-Tuning')
    parser.add_argument('--epochs', type=int, help='Number of Training Epochs')
    parser.add_argument('--cd-size', type=float, default=0.0, metavar='CD',
                        help='complexity decay size (default: 0.0)')
    parser.add_argument('--cd-ops', type=float, default=0.0, metavar='CD',
                        help='complexity decay ops (default: 0.0)')
    parser.add_argument('--size-target', type=float, default=0, metavar='ST',
                        help='target size (default: 0)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Directory with Training Data')
    args = parser.parse_args()
    main(args)
