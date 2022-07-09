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

#setting_getter = argparse.ArgumentParser(description='Put your experiment file and version number')
#setting_getter.add_argument('--text_path', '-t', type=str, help='your experiment file')
#setting_getter.add_argument('--version', '-v', type=int, help='version number to experiment')

#setting_getter = setting_getter.parse_args()
#txt_lst = get_version_text(setting_getter.version, setting_getter.text_path)

#args = Version_Dictionary(setting_getter.version, txt_lst)
#args.set_value()
parser = argparse.ArgumentParser(description='Put your wanted GPU num(zero indexed) and selected GPU vram size')
parser.add_argument('--number', '-n', type=str, help='GPU_number, zero indexed')

os.environ["CUDA_VISIBLE_DEVICES"]=parser.number
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


