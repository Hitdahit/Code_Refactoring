import os
from turtle import window_width
import cv2
import glob
import pydicom
import numpy as np

import albumentations as A
import albumentations.pytorch

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import util_3D

# Make Dataset
class CT_Dataset(Dataset):

    def _BC_labeler(self):
        '''
        get label from path
        '''
        label = [[0, 1], [1, 0]]
        imgs = [sorted(os.listdir(os.path.join(self.img_dir, i))) for i in self.classes]
        labels = np.array([label[idx] for idx, i in enumerate(imgs) for j in i], dtype=np.float32)
        imgs = [os.path.join(self.img_dir, self.classes[idx], j) for idx, i in enumerate(imgs) for j in i]
        
        return imgs, labels
    def _MC_labeler(self):
        '''
        get label from filename
        '''
        pass
    def _ML_labeler(self):
        '''
        custom your labeler
        '''
        pass

    def __init__(self, img_dir, mode, image_size, window_width, window_level, normalization, transform):
        '''
        data's root dir
        '''
        self.img_dir = os.path.join(img_dir, mode)
        self.image_size = image_size # Get image_size    
        self.window_width = window_width # Get window_width
        self.window_level = window_level # Get window_level
        self.normalization = normalization # Get normalization method
        self.transform = transform # Get augmentation

        self.classes = sorted(os.listdir(self.img_dir))
        
        imgs, labels = self._BC_labeler()
        self.imgs = imgs
        self.labels = labels      

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        image_size = self.image_size # Get image_size
        window_width = self.window_width # Get window_width
        window_level = self.window_level # Get window_level
        normalization = self.normalization
        
        # Pre-process task
        img = util_3D.CT_preprocess(img, image_size, window_width, window_level, normalization) # Data value range : 0.0 ~ 1.0
        albu_dic = self.transform(image=img) # Apply augmentation to data

        data_dic= {'image' : albu_dic['image'], 'label' : label} # Make dictionary which has one data to image key and one label to label key
        return data_dic
    

# get loader from Dataset as batch size
# img_dir, label_dir, preprocess_type, transform, batch_size, workers
def get_loader(args, modality, mode='train'):
    if modality == 'CT':
        dataset = CT_Dataset(args.data_dir, mode, args.img_size, args.window_width, args.window_level, args.normalization, args.augmentation)
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                shuffle=args.shuffle, num_workers=args.num_worker,
                                drop_last=args.drop_last)
    elif modality == 'MR':
        '''
        will be updated...
        '''
    elif modality == 'Endo':
        '''
        will be updated...
        '''
    return dataloader


