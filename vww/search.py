import argparse
import copy
import sys
sys.path.append('..')

import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from data_wrapper import VWWDataWrapper
import models as models

# Simply parse all models' names contained in models directory
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Visual Wake Words Search')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='plain_mobilenetv1',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: plain_mobilenetv1)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=70, metavar='N',
                        help='number of epochs to train (default: 70')
    parser.add_argument('--early-stop', type=int, default=None,
                        help='Early-Stop patience (default: None)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lra', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
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
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
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
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Data preparation
    data_dir = 'data/vw_coco2014_96'
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        validation_split=0.1,
        rescale=1./255)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size = (96, 96),
        batch_size = args.batch_size,
        subset = 'training',
        color_mode = 'rgb'
        )
    test_generator = datagen.flow_from_directory(
        data_dir,
        target_size = (96, 96),
        batch_size = args.batch_size,
        subset = 'validation',
        color_mode = 'rgb'
        )

    train_set = VWWDataWrapper(data_generator=train_generator)
    # Split dataset into train and validation
    train_len = int(len(train_set) * 0.9)
    val_len = len(train_set) - train_len
    # Fix generator seed for reproducibility
    data_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_len, val_len], generator=data_gen)

    test_set = VWWDataWrapper(data_generator=test_generator)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=None, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=None, shuffle=False,
        num_workers=4, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=None, shuffle=False,
        num_workers=4, pin_memory=True)

    # Build model, optimizer and lr scheduler
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]().to(device)

    # Check if pretrained model exists
    if args.pretrained_model is not None:
        print("=> using pre-trained model '{}'".format(args.pretrained_model))
        model.load_state_dict(torch.load(args.pretrained_model))
    else:
        print("=> no pre-trained model found")

    # Group model/architecture parameters
    params, alpha_params = [], []
    for name, param in model.named_parameters():
        if 'alpha' in name:
            alpha_params += [param]
        else:
            params += [param]

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    #arch_optimizer = optim.SGD(alpha_params, lr=args.lra, momentum=0.9)
    arch_optimizer = optim.Adam(alpha_params, lr=args.lra)

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
        train(args, model, device, train_loader, optimizer, arch_optimizer, epoch)
        val_acc = test(model, device, val_loader, scope='Val')
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
        adjust_learning_rate(optimizer, epoch, args)
    
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

def adjust_learning_rate(optimizer, epoch, args):
    if epoch == 21:
        args.lr = args.lr * 0.5
    elif epoch == 31:
        args.lr = args.lr * 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

def train(args, model, device, train_loader, optimizer, arch_optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        arch_optimizer.zero_grad()
        output = model(data.transpose(1,3).transpose(2,3))
        loss = nn.BCEWithLogitsLoss()(output, target)
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
        arch_optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSize-Loss: {:.6f}\tOps-Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), loss_size.item(), loss_ops.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, scope='Test'):
    model.eval()
    if scope == 'Val':
        samples = 9866
    else:
        samples = 10961
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.transpose(1,3).transpose(2,3))
            test_loss += nn.BCEWithLogitsLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        scope, test_loss, correct, samples,
        100. * correct / samples))

    return 100. * correct / samples

if __name__ == '__main__':
    main()