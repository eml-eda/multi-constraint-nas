import argparse
import pathlib
from typing import Dict
import math

from pytorch_model_summary import summary
from torchinfo import summary as summ
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from flexnas.methods import PIT
from flexnas.methods import PITSuperNet
from flexnas.methods.pit.nn import PITConv2d, PITLinear
from flexnas.methods.pit_supernet.nn import PITSuperNetCombiner
import pytorch_benchmarks.image_classification as icl
from pytorch_benchmarks.utils import AverageMeter, seed_all, accuracy, CheckPoint, EarlyStopping

from utils import evaluate, train_one_epoch
from hardware_model import get_latency_conv2D_GAP8, get_latency_Linear_GAP8, get_latency_conv2D_Diana, get_latency_Linear_Diana, get_latency_total
from hardware_model import compute_layer_latency_GAP8, compute_layer_latency_Diana, get_latency_binarized_supernet, get_size_binarized_supernet
from models import ResNet8PITSN


def main(args):
    DATA_DIR = None
    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datasets = icl.get_data(data_dir=DATA_DIR)
    input_shape = datasets[0][0][0].numpy().shape
    # Get the Model
    if args.model == "PIT":
        model = icl.get_reference_model('resnet_8')
        model = model.to(device)
        pit_model = PIT(model, input_shape=input_shape)
        pit_model = pit_model.to(device)
        pit_model.train_features = True
        pit_model.train_rf = False
        pit_model.train_dilation = False
    elif args.model == "Supernet":
        model = ResNet8PITSN()
        model = model.to(device)
        pit_model = PITSuperNet(model, input_shape=input_shape, autoconvert_layers = False)
        pit_model = pit_model.to(device)
    
    param_dicts = [
        {'params': pit_model.nas_parameters(), 'weight_decay': 0},
        {'params': pit_model.net_parameters()}]
    optimizer = torch.optim.Adam(param_dicts, lr=0.001, weight_decay=1e-4)
    search_checkpoint = CheckPoint('./search_checkpoints', pit_model, optimizer, 'max', fmt='prova.pt')
    search_checkpoint.load(f'./search_checkpoints/ck_icl_opt_Supernet_max_targets_mem_increasing_size_20000.0_lat_2000000.0_063.pt')
    print("Alpha: " + str(pit_model.seed.inputblock.conv1.sn_combiner.alpha))
    print("Temperature: " + str(pit_model.seed.inputblock.conv1.sn_combiner.softmax_temperature*pow(math.exp(-0.1),63)))
    print("Alpha/Temperature: " + str(nn.functional.softmax(pit_model.seed.inputblock.conv1.sn_combiner.alpha/(pit_model.seed.inputblock.conv1.sn_combiner.softmax_temperature*pow(math.exp(-0.1),63)),dim=0)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS Search and Fine-Tuning')
    parser.add_argument('--model', type=str, default="const",
                        help='PIT, Supernet')
    args = parser.parse_args()
    main(args)
