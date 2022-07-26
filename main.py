# !pip install addict
# !pip install yapf
# !pip install seaborn
# !pip install pydicom

import os
import numpy as np
import parser

import torch

from train import *
import train
import argparse
from configs_wrapper import setting
from data_loader import Dataset
from torch.utils.data import DataLoader

import os
import configs
import sys
import models

parser = argparse.ArgumentParser(description='Put your wanted GPU num(zero indexed) and selected GPU vram size')
parser.add_argument('--number', '-n', type=str, help='GPU_number, zero indexed')
parser = parser.parse_args()

print('gpu? ', torch.cuda.is_available())
device = torch.device(f'cuda:{parser.number}' if torch.cuda.is_available() else 'cpu')
# device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print('Current gpu: ', torch.cuda.current_device())

dic = configs.Config.fromfile('./experiments.py')
args = setting(dic)
args.parse()

setattr(args, 'train_dataloader', DataLoader(Dataset(args, mode='train'), 
                                             batch_size= args.batch_size, shuffle=True, num_workers=0))
setattr(args, 'valid_dataloader', DataLoader(Dataset(args, mode='valid'), 
                                             batch_size= args.batch_size, shuffle=True, num_workers=0))
setattr(args, 'device', device)

for i in range(args.epoch):
    train_res = train_one_epoch(args, i)
    val_res = valid_one_epoch(args, i)


