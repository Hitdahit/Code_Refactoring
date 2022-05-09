import os
import numpy as np
import parser

import torch
from version_parser import get_version_text, Version_Dictionary
from train import *
import argparse


setting_getter = argparse.ArgumentParser(description='Put your experiment file and version number')
setting_getter.add_argument('--text_path', '-t', type=str, help='your experiment file')
setting_getter.add_argument('--version', '-v', type=int, help='version number to experiment')

setting_getter = setting_getter.parse_args()
txt_lst = get_version_text(setting_getter.version, setting_getter.text_path)

args = Version_Dictionary(setting_getter.version, txt_lst)
args.set_value()


'''
TODO: configs 나머지 코드에 녹이기
from utils import configs
setting_dict, setting_txt = configs.Config.fromfile('./settings.py')

'''


if 'BC' in args.task_type:
    train_loss, train_metric, val_loss, val_metric = binary_classification_train(args)
elif 'MC' in args.task_type:
    multiclass_classification_train(args)
elif 'ML' in args.task_type:
    pass
