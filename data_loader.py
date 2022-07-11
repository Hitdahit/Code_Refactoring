import os
import cv2
import glob
import pydicom
import numpy as np

import albumentations as A
import albumentations.pytorch

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


# Make Dataset
class Dataset(Dataset):
    def __init__(self, args, mode='train'):
        '''
        data's root dir
        '''
        self.img_dir = args.data_root
        self.image_size = args.img_size
        self.labeler = args.labeler
        self.modality = args.modality    
        self.preprocessor = args.prep_config    
        self.classes = args.classes
        
        self.transform = args.train_augmentations if 'train' in mode else args.valid_augmentations
        
        imgs = [sorted(os.listdir(os.path.join(self.img_dir, mode, i))) for i in self.classes]
        self.imgs = [os.path.join(self.img_dir, mode, self.classes[idx], j) for idx, i in enumerate(imgs) for j in i]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        if 'EN' in self.modality:
            img = cv2.imread(self.imgs[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = pydicom.dcmread(self.imgs[idx])

        img = self.preprocessor.execute(img)
        label = self.labeler.execute(self.imgs[idx])
        
        albu_dic = self.transform(image=img)

        data_dic= {'image' : albu_dic['image'], 'label' : label} # Make dictionary which has one data to image key and one label to label key
        return data_dic
    


'''
# get loader from Dataset as batch size
# img_dir, label_dir, preprocess_type, transform, batch_size, workers
def get_loader(args, mode='train'):
    dataset = Dataset(args.data_dir, mode, args.img_size, args.modality, args.augmentation,
                      args.window_width, args.window_level, args.photometricInterpretation, args.normalize_range, args.percentage)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            shuffle=args.shuffle, num_workers=args.num_worker,
                            drop_last=args.drop_last)
    return dataloader


'''