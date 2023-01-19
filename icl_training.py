import argparse
import pathlib
from typing import Dict

from pytorch_model_summary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from flexnas.methods import PIT
import pytorch_benchmarks.image_classification as icl
from pytorch_benchmarks.utils import AverageMeter, seed_all, accuracy, CheckPoint, EarlyStopping

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class ResNet8(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Resnet v1 parameters
        self.input_shape = [3, 32, 32]  # default size for cifar10
        self.num_classes = 10  # default class number for cifar10
        self.num_filters = 16  # this should be 64 for an official resnet model

        # Resnet v1 layers

        # First stack
        self.inputblock = ConvBlock(in_channels=3, out_channels=16,
                                    kernel_size=3, stride=1, padding=1)
        self.convblock1 = ConvBlock(in_channels=16, out_channels=16,
                                    kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # Second stack
        self.convblock2 = ConvBlock(in_channels=16, out_channels=32,
                                    kernel_size=3, stride=2, padding=1)
        self.conv2y = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2y.weight)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2x = nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0)
        nn.init.kaiming_normal_(self.conv2x.weight)

        # Third stack
        self.convblock3 = ConvBlock(in_channels=32, out_channels=64,
                                    kernel_size=3, stride=2, padding=1)
        self.conv3y = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv3y.weight)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3x = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)
        nn.init.kaiming_normal_(self.conv3x.weight)

        self.avgpool = torch.nn.AvgPool2d(8)

        self.out = nn.Linear(64, 10)
        nn.init.kaiming_normal_(self.out.weight)

    def forward(self, input):
        # Input layer
        x = self.inputblock(input)  # [32, 32, 16]

        # First stack
        y = self.convblock1(x)      # [32, 32, 16]
        y = self.conv1(y)
        y = self.bn1(y)
        x = torch.add(x, y)         # [32, 32, 16]
        x = self.relu(x)

        # Second stack
        y = self.convblock2(x)      # [16, 16, 32]
        y = self.conv2y(y)
        y = self.bn2(y)
        x = self.conv2x(x)          # [16, 16, 32]
        x = torch.add(x, y)         # [16, 16, 32]
        x = self.relu(x)

        # Third stack
        y = self.convblock3(x)      # [8, 8, 64]
        y = self.conv3y(y)
        y = self.bn3(y)
        x = self.conv3x(x)          # [8, 8, 64]
        x = torch.add(x, y)         # [8, 8, 64]
        x = self.relu(x)

        x = self.avgpool(x)         # [1, 1, 64]
        # x = torch.squeeze(x)        # [64]
        x = torch.flatten(x, 1)
        x = self.out(x)             # [10]

        return x

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
        args) -> Dict[str, float]:
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
        val_metrics = evaluate(search, model, criterion, val_dl, device)
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

    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    # Ensure determinstic execution
    seed_all(seed=14)

    # Get the Data
    # data_dir = os.path.join(os.getcwd(), DATA_DIR)
    data_dir = DATA_DIR
    datasets = icl.get_data(data_dir=data_dir)
    dataloaders = icl.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders

    # Get the Model
    model = ResNet8()
    model = model.to(device)

    # Model Summary
    input_example = torch.unsqueeze(datasets[0][0][0], 0).to(device)
    input_shape = datasets[0][0][0].numpy().shape
    print(summary(model, input_example, show_input=False, show_hierarchical=True))

    # Warmup Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    warmup_checkpoint = CheckPoint('./warmup_checkpoints', model, optimizer, 'max',fmt='ck_icl_{epoch:03d}.pt')
    skip_warmup = True
    if pathlib.Path('./warmup_checkpoints/final_best_warmup_icl.ckp').exists():
        warmup_checkpoint.load('./warmup_checkpoints/final_best_warmup_icl.ckp')
        print("Skipping warmup")
    else:
        skip_warmup = False
        print("Running warmup")

    if not skip_warmup:
        for epoch in range(N_EPOCHS):
            metrics = train_one_epoch(
                epoch, False, model, criterion, optimizer, train_dl, val_dl, test_dl, device, args)
            scheduler.step()
            warmup_checkpoint(epoch, metrics['val_acc'])
        warmup_checkpoint.load_best()
        warmup_checkpoint.save('./warmup_checkpoints/final_best_warmup_icl.ckp')

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
    criterion = icl.get_default_criterion()
    param_dicts = [
        {'params': pit_model.nas_parameters(), 'weight_decay': 0},
        {'params': pit_model.net_parameters()}]
    optimizer = torch.optim.Adam(param_dicts, lr=0.001, weight_decay=1e-4)
    scheduler = icl.get_default_scheduler(optimizer)
    # Set EarlyStop with a patience of 20 epochs and CheckPoint
    earlystop = EarlyStopping(patience=20, mode='max')
    search_checkpoint = CheckPoint('./search_checkpoints', pit_model, optimizer, 'max', fmt='ck_icl_{epoch:03d}.pt')
    for epoch in range(N_EPOCHS):
        metrics = train_one_epoch(
            epoch, True, pit_model, criterion, optimizer, train_dl, val_dl, test_dl, device, args)

        if epoch > 0:
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
    criterion = icl.get_default_criterion()
    optimizer = icl.get_default_optimizer(exported_model)
    scheduler = icl.get_default_scheduler(optimizer)
    finetune_checkpoint = CheckPoint('./finetuning_checkpoints', pit_model, optimizer, 'max', fmt='ck_icl_{epoch:03d}.pt')
    earlystop = EarlyStopping(patience=50, mode='max')
    for epoch in range(N_EPOCHS):
        metrics = train_one_epoch(
            epoch, False, exported_model, criterion, optimizer, train_dl, val_dl, test_dl, device, args)
        scheduler.step()
        if epoch > 0:
            finetune_checkpoint(epoch, metrics['val_acc'])
            if earlystop(metrics['val_acc']):
                break
    test_metrics = evaluate(False, exported_model, criterion, test_dl, device)
    print("Fine-tuning Test Set Loss:", test_metrics['loss'])
    print("Fine-tuning Test Set Accuracy:", test_metrics['acc'])


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
