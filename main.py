import os
import numpy as np
import parser

import torch
from version_parser import get_version_text, Version_Dictionary, parser
from train import *


'''
parser 구현! txt 파일경로와 version번호 받아오는 거
'''
txt_lst = get_version_text(version, path)

args = Version_Dictionary(txt_lst)

args = parser(args)

if 'BC' in args.task_type:
    train_loss, train_metric, val_loss, val_metric = binary_classification_train(args)
elif 'MC' in args.task_type:
    multiclass_classification_train(args)
elif 'ML' in args.task_type:
    pass
