import os
import numpy as np
import parser

import torch

from train import *
import argparse
from configs_wrapper import setting
from data_loader import Dataset

import os
import configs
import sys
import models

parser = argparse.ArgumentParser(description='Put your wanted GPU num(zero indexed) and selected GPU vram size')
parser.add_argument('--number', '-n', type=str, help='GPU_number, zero indexed')
parser = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = parser.number
print('gpu? ', torch.cuda.is_available())
device = torch.device(f'cuda:{parser.number}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print('Current gpu: ', torch.cuda.current_device())



dic = configs.Config.fromfile('./experiments.py')
args = setting(dic)
args.parse()

setattr(args, 'dataloader', Dataset(args))
setattr(args, 'device', device)

for i in range(args.epoch):
    train_res = train_one_epoch(args, i)
    val_res = valid_one_epoch(args, i)


