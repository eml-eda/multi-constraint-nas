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
    parser = argparse.ArgumentParser(description='PyTorch Visual Wake Words Fine-Tuning')
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
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
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
    parser.add_argument('--found-model', type=str, default=None,
                        help='path where the searched model is stored')
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
    test_set = VWWDataWrapper(data_generator=test_generator)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=None, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=None, shuffle=False,
        num_workers=4, pin_memory=True)
    
    # Build model, optimizer and lr scheduler
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](found_model=args.found_model, ft=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)

    # Training
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        adjust_learning_rate(optimizer, epoch, args)
    
    # Save model    
    torch.save(model.state_dict(), 
        f"saved_models/ft_{args.arch}_target-{args.size_target:.1e}_cdops-{args.cd_ops:.1e}.pth.tar")

def adjust_learning_rate(optimizer, epoch, args):
    if epoch == 21:
        args.lr = args.lr * 0.5
    elif epoch == 31:
        args.lr = args.lr * 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.transpose(1,3).transpose(2,3))
        loss = nn.BCEWithLogitsLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, samples,
        100. * correct / samples))

if __name__ == '__main__':
    main()