import argparse
import pathlib
from typing import Dict

from pytorch_model_summary import summary
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from flexnas.methods import PIT
import pytorch_benchmarks.keyword_spotting as kws
from pytorch_benchmarks.utils import AverageMeter, seed_all, accuracy, CheckPoint, EarlyStopping


# Definition of evaluation function
def evaluate(
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device,
        reg_strength: torch.Tensor = torch.tensor(0.)) -> Dict[str, float]:
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
            if search:
                loss_reg = reg_strength * model.get_regularization_loss()
                loss = loss_task + loss_reg
            else:
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
        reg_strength: torch.Tensor = torch.tensor(0.)) -> Dict[str, float]:
    model.train()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0
    with tqdm(total=len(train_dl), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for audio, target in train_dl:
            step += 1
            tepoch.update(1)
            audio, target = audio.to(device), target.to(device)
            output = model(audio)
            loss_task = criterion(output, target)
            if search:
                loss_reg = reg_strength * model.get_regularization_loss()
                loss = loss_task + loss_reg
            else:
                loss = loss_task
                loss_reg = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], audio.size(0))
            avgloss.update(loss, audio.size(0))
            avglosstask.update(loss_task, audio.size(0))
            avglossreg.update(loss_reg, audio.size(0))
            if step % 100 == 99:
                tepoch.set_postfix({'loss': avgloss,
                                    'loss_task': avglosstask,
                                    'loss_reg': avglossreg,
                                    'acc': avgacc})
        val_metrics = evaluate(search, model, criterion, val_dl, device, reg_strength)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        test_metrics = evaluate(search, model, criterion, test_dl, device)
        test_metrics = {'test_' + k: v for k, v in test_metrics.items()}
        final_metrics = {
            'loss': avgloss.get(),
            'loss_task': avglosstask.get(),
            'loss_reg': avglossreg.get(),
            'acc': avgacc.get(),
        }
        final_metrics.update(val_metrics)
        tepoch.set_postfix(final_metrics)
        tepoch.close()
        print(val_metrics)
        print(test_metrics)
        return final_metrics


def main(args):
    DATA_DIR = args.data_dir
    N_EPOCHS = args.epochs
    LAMBDA = torch.tensor(args.strength)
    CKP_DIR = args.ckp_dir

    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    # Ensure determinstic execution
    seed_all(seed=14)

    # Get the Data
    data_dir = DATA_DIR
    datasets = kws.get_data(data_dir=data_dir)
    dataloaders = kws.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders

    # Get the Model
    model = kws.get_reference_model('ds_cnn')
    model = model.to(device)

    # Model Summary
    input_example = torch.unsqueeze(datasets[0][0][0], 0).to(device)
    input_shape = datasets[0][0][0].numpy().shape
    print(summary(model, input_example, show_input=False, show_hierarchical=True))

    # Warmup Loop
    criterion = kws.get_default_criterion()
    optimizer = kws.get_default_optimizer(model)
    scheduler = kws.get_default_scheduler(optimizer)
    warmup_checkpoint = CheckPoint(f'.{CKP_DIR}/warmup_checkpoints', model, optimizer, 'max')
    skip_warmup = True
    if pathlib.Path(f'.{CKP_DIR}/final_best_warmup.ckp').exists():
        warmup_checkpoint.load(f'.{CKP_DIR}/final_best_warmup.ckp')
        print("Skipping warmup")
    else:
        skip_warmup = False
        print("Running warmup")

    if not skip_warmup:
        for epoch in range(N_EPOCHS):
            metrics = train_one_epoch(
                epoch, False, model, criterion, optimizer, train_dl, val_dl, test_dl, device)
            scheduler.step()
            warmup_checkpoint(epoch, metrics['val_acc'])
        warmup_checkpoint.load_best()
        warmup_checkpoint.save(f'.{CKP_DIR}/final_best_warmup.ckp')

    test_metrics = evaluate(False, model, criterion, test_dl, device)
    print("Warmup Test Set Loss:", test_metrics['loss'])
    print("Warmup Test Set Accuracy:", test_metrics['acc'])

    # Convert the model to PIT
    pit_model = PIT(model, input_shape=input_shape)
    pit_model = pit_model.to(device)
    pit_model.train_features = True
    pit_model.train_rf = False
    pit_model.train_dilation = False
    print(summary(pit_model, input_example, show_input=False, show_hierarchical=True))

    # Search Loop
    criterion = kws.get_default_criterion()
    param_dicts = [
        {'params': pit_model.nas_parameters(), 'weight_decay': 0},
        {'params': pit_model.net_parameters()}]
    optimizer = torch.optim.Adam(param_dicts, lr=0.0005, weight_decay=1e-4)
    scheduler = kws.get_default_scheduler(optimizer)
    # Set EarlyStop with a patience of 20 epochs and CheckPoint
    earlystop = EarlyStopping(patience=20, mode='max')
    search_checkpoint = CheckPoint(f'.{CKP_DIR}/search_checkpoints', pit_model, optimizer, 'max')
    for epoch in range(N_EPOCHS):
        metrics = train_one_epoch(
            epoch, True, pit_model, criterion, optimizer, train_dl, val_dl, test_dl, device,
            reg_strength=LAMBDA)

        if epoch > 5:
            search_checkpoint(epoch, metrics['val_acc'])
            if earlystop(metrics['val_acc']):
                break

        scheduler.step()
        print("architectural summary:")
        print(pit_model)
        print("model size:", pit_model.get_size())
    print("Load best model")
    search_checkpoint.load_best()
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
    criterion = kws.get_default_criterion()
    optimizer = kws.get_default_optimizer(exported_model)
    scheduler = kws.get_default_scheduler(optimizer)
    for epoch in range(N_EPOCHS):
        train_one_epoch(
            epoch, False, exported_model, criterion, optimizer, train_dl, val_dl, test_dl, device)
        scheduler.step()
    test_metrics = evaluate(False, exported_model, criterion, test_dl, device)
    print("Fine-tuning Test Set Loss:", test_metrics['loss'])
    print("Fine-tuning Test Set Accuracy:", test_metrics['acc'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS Search and Fine-Tuning')
    parser.add_argument('--strength', type=float, help='Regularization Strength')
    parser.add_argument('--epochs', type=int, help='Number of Training Epochs')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Directory with Training Data')
    parser.add_argument('--ckp-dir', type=str, default="",
                        help='Path to Directory with Checkpoints')
    args = parser.parse_args()
    main(args)
