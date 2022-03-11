import argparse
import copy
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

import models as models

# Simply parse all models' names contained in models directory
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Search')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='plain_resnet8',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: plain_resnet8)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--early-stop', type=int, default=None,
                        help='Early-Stop patience (default: None)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
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
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path to a pretrained model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    trainset = datasets.CIFAR10('data', train=True, download=True,
                       transform=transform_train)
    # Split dataset into train and validation
    train_len = int(len(trainset) * 0.9)
    val_len = len(trainset) - train_len
    # Fix generator seed for reproducibility
    data_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_len, val_len], generator=data_gen)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    testset = datasets.CIFAR10('data', train=False, download=True,
                       transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]().to(device)

    # Check if pretrained model exists
    if args.pretrained_model is not None:
        print("=> using pre-trained model '{}'".format(args.pretrained_model))
        model.load_state_dict(torch.load(args.pretrained_model))
    else:
        print("=> no pre-trained model found")

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Dummy test step to get inital complexities of model
    test(model, device, test_loader, scope='Initial Dummy Test')
    size_i = sum(model.size_dict.values()).clone().detach().cpu().numpy()
    ops_i = sum(model.ops_dict.values()).clone().detach().cpu().numpy()
    print(f"Initial size: {size_i:.3e} params\tInitial ops: {ops_i:.3e} OPs")
    alive_ch_i = copy.deepcopy(model.alive_ch)
    for k, v in alive_ch_i.items():
        print(f"{k}:\t{int(v)+1} channels")
    if args.eval_complexity:
        print("Exit...")
        return

    # Training
    val_acc = 0
    val_acc_best = 0
    epoch_wout_improve = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        val_acc = test(model, device, val_loader, scope='Validation')
        if args.early_stop is not None:
            if val_acc > val_acc_best and epoch >= 10:
                val_acc_best = val_acc
                epoch_wout_improve = 0
                # Save model
                print("=> saving new best model")
                torch.save(model.state_dict(), 
                    f"saved_models/srch_{args.arch}_target-{args.size_target:.1e}_cdops-{args.cd_ops:.1e}.pth.tar")
            else:
                epoch_wout_improve += 1 if epoch > 10 else 0
                print(f"No improvement in {epoch_wout_improve} epochs")
                print(f"Keep going for {args.early_stop - epoch_wout_improve} epochs")
        test(model, device, test_loader, scope='Test')
        scheduler.step()
    
        # Compute and print final size and ops
        size_f = sum(model.size_dict.values()).clone().detach().cpu().numpy()
        ops_f = sum(model.ops_dict.values()).clone().detach().cpu().numpy()
        print(f"Final size: {size_f:.3e}/{size_i:.3e} parameters\tFinal ops: {ops_f:.3e}/{ops_i:.3e} OPs")
        # Print learned alive channels
        for k, v in model.alive_ch.items():
            print(f"{k}:\t{int(v)+1}/{int(alive_ch_i[k])+1} channels")
        
        if args.early_stop is not None: 
            if epoch_wout_improve >= args.early_stop:
                print("Early stopping...")
                break

    # Save model
    if args.early_stop is None:
        torch.save(model.state_dict(), 
            f"saved_models/srch_{args.arch}_target-{args.size_target:.1e}_cdops-{args.cd_ops:.1e}.pth.tar")

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
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
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        scope, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

if __name__ == '__main__':
    main()
