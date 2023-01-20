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

                # Compute size-complexity loss with constraint
                if args.loss_type == "abs":
                    if "mem" in args.loss_elements:
                        loss_reg = cd_size * torch.abs((model.get_size() - args.size_target))
                    else:
                        loss_reg = cd_size * model.get_size()
                    if "lat" in args.loss_elements:
                        loss_ops = cd_ops * torch.abs((model.get_macs() - args.latency_target))
                    else:
                        loss_ops = cd_ops * model.get_macs()

                elif args.loss_type == "max":
                    if "mem" in args.loss_elements:
                        loss_reg = cd_size * torch.max((model.get_size() - args.size_target), torch.FloatTensor([0]))[0]
                    else:
                        loss_reg = cd_size * model.get_size()
                    if "lat" in args.loss_elements:
                        loss_ops = cd_ops * torch.max((model.get_macs() - args.latency_target), torch.FloatTensor([0]))[0]
                    else:
                        loss_ops = cd_ops * model.get_macs()

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
    model = icl.get_reference_model('resnet_8')
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
    name = f"ck_icl_opt_{args.loss_type}_targets_{args.loss_elements}_size_{args.size_target}_lat_{args.latency_target}"
    search_checkpoint = CheckPoint('./search_checkpoints', pit_model, optimizer, 'max', fmt=name+'_{epoch:03d}.pt')
    print("Initial model size:", pit_model.get_size())
    print("Initial model MACs:", pit_model.get_macs())
    increment_cd_size = (args.cd_size*99/100)/int(args.epochs/5)
    increment_cd_ops = (args.cd_ops*99/100)/int(args.epochs/5)
    for epoch in range(N_EPOCHS):
        metrics = train_one_epoch(
            epoch, True, pit_model, criterion, optimizer, train_dl, val_dl, test_dl, device, args, increment_cd_size, increment_cd_ops)


        if epoch > 0:
            search_checkpoint(epoch, metrics['val_acc'])
            if earlystop(metrics['val_acc']):
                break

        scheduler.step()
        print("architectural summary:")
        print(pit_model)
        print("model size:", pit_model.get_size())
        print("model MACs:", pit_model.get_macs())
        print(f"cd_size:  {min(args.cd_size/100 + increment_cd_size*epoch, args.cd_size)} cd_ops: {min(args.cd_ops/100 + increment_cd_ops*epoch, args.cd_ops)}")
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
    parser.add_argument('--latency-target', type=float, default=0, metavar='ST',
                        help='target latency (default: 0)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Directory with Training Data')
    parser.add_argument('--loss_type', type=str, default="max",
                        help='abs, max')
    parser.add_argument('--loss_elements', type=str, default="mem",
                        help='loss type: mem, lat, mem+lat')
    parser.add_argument('--l', type=str, default="const",
                        help='const, increasing')
    args = parser.parse_args()
    main(args)
