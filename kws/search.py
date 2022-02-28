import argparse
import copy
import sys
sys.path.append('..')

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from data_wrapper import KWSDataWrapper
import get_dataset as kws_data
import kws_util
import models as models

# Simply parse all models' names contained in models directory
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Keyword Spotting Search')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='plain_dscnn',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: plain_dscnn)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=90, metavar='N',
                        help='number of epochs to train (default: 90)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--cd-size', type=float, default=0.0, metavar='CD',
                        help='complexity decay size (default: 0.0)')
    parser.add_argument('--cd-ops', type=float, default=0.0, metavar='CD',
                        help='complexity decay ops (default: 0.0)')
    parser.add_argument('--size-target', type=float, default=0, metavar='ST',
                        help='target size (default: 0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--eval-complexity', action='store_true', default=False,
                        help='Evaluate complexity of initial model and exit')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Data download and pre-processing
    data_dir = 'data/'
    Flags, unparsed = kws_util.parse_command()
    Flags.data_dir = data_dir
    Flags.bg_path = data_dir
    print(f'We will download data to {Flags.data_dir}')
    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
    print("Done getting data")

    # Train Set
    train_shuffle_buffer_size = 85511
    ds_train = list(ds_train.shuffle(train_shuffle_buffer_size).as_numpy_iterator())
    x_train, y_train = [], []
    for x, y in ds_train:
        x_train.append(x)
        y_train.append(np.expand_dims(y, axis=1))
    x_train = np.vstack(x_train)
    y_train = np.vstack(y_train).squeeze(-1)

    train_set = KWSDataWrapper(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, **cuda_kwargs)

    # Val Set
    val_shuffle_buffer_size = 10102
    ds_val = list(ds_val.shuffle(val_shuffle_buffer_size).as_numpy_iterator())
    x_val, y_val = [], []
    for x, y in ds_val:
        x_val.append(x)
        y_val.append(np.expand_dims(y, axis=1))
    x_val = np.vstack(x_val)
    y_val = np.vstack(y_val).squeeze(-1)

    val_set = KWSDataWrapper(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.test_batch_size, **cuda_kwargs)

    # Test Set
    test_shuffle_buffer_size = 4890
    ds_test = list(ds_test.shuffle(test_shuffle_buffer_size).as_numpy_iterator())
    x_test, y_test = [], []
    for x, y in ds_test:
        x_test.append(x)
        y_test.append(np.expand_dims(y, axis=1))
    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test).squeeze(-1)

    test_set = KWSDataWrapper(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.test_batch_size, **cuda_kwargs)

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)

    # Dummy test step to get inital complexities of model
    test(model, device, test_loader)
    size_i = sum(model.size_dict.values()).clone().detach().cpu().numpy()
    ops_i = sum(model.ops_dict.values()).clone().detach().cpu().numpy()
    print(f"Initial size: {size_i:.3e} params\tInitial ops: {ops_i:.3e} OPs")
    alive_ch_i = copy.deepcopy(model.alive_ch)
    for k, v in alive_ch_i.items():
        print(f"{k}:\t{int(v)+1} channels")
    if args.eval_complexity:
        print("Exit...")
        return

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, scope='Validation')
        test(model, device, test_loader, scope='Test')
        adjust_learning_rate(optimizer, epoch)
    
    # Compute and print final size and ops
    size_f = sum(model.size_dict.values()).clone().detach().cpu().numpy()
    ops_f = sum(model.ops_dict.values()).clone().detach().cpu().numpy()
    print(f"Final size: {size_f:.3e}/{size_i:.3e} parameters\tFinal ops: {ops_f:.3e}/{ops_i:.3e} OPs")
    # Print learned alive channels
    for k, v in model.alive_ch.items():
        print(f"{k}:\t{int(v)+1}/{int(alive_ch_i[k])+1} channels")

    # Save model
    torch.save(model.state_dict(), 
        f"saved_models/srch_{args.arch}_target-{args.size_target:.1e}_cdops-{args.cd_ops:.1e}.pth.tar")

def adjust_learning_rate(optimizer, epoch):
    if epoch < 50:
        lr = 1e-2
    elif epoch < 100:
        lr = 5e-3
    elif epoch < 150:
        lr = 2.5e-3
    else:
        lr = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).squeeze(0), target.to(device).squeeze(0)
        optimizer.zero_grad()

        # compute output
        output = model(data.transpose(1,3).transpose(2,3))
        loss = nn.CrossEntropyLoss()(output, target)

        # Compute size-complexity loss with constraint
        loss_size = torch.abs(sum(model.size_dict.values()) - args.size_target)
        loss_size *= args.cd_size
        # Compute ops-complexity loss with constraint
        loss_ops = sum(model.ops_dict.values())
        loss_ops *= args.cd_ops
        # Sum different losses
        loss += loss_size + loss_ops
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSize-Loss: {:.6f}\tOps-Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), loss_size.item(), loss_ops.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, scope='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).squeeze(0), target.to(device).squeeze(0)
            output = model(data.transpose(1,3).transpose(2,3))
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        scope, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    main()
