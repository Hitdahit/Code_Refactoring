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
import util

# Make Dataset
class Dataset(Dataset):

    def _BC_labeler(self):
        '''
        get label from path
        '''
        label = [[0, 1], [1, 0]]
        imgs = [sorted(os.listdir(os.path.join(self.data_dir, i))) for i in self.classes]
        labels = np.array([label[idx] for idx, i in enumerate(self.imgs) for j in i], dtype=np.float32)
        imgs = [os.path.join(self.data_dir, self.classes[idx], j) for idx, i in enumerate(self.imgs) for j in i]
        
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

    def __init__(self, img_dir, image_size, modality, transform):
        '''
        data's root dir
        '''
        self.img_dir = img_dir
        self.image_size = image_size # Get image_size
        self.modality = modality        
        self.transform = transform # Get augmentation

        self.classes =sorted( os.listdir(self.img_dir))
        
        imgs, labels = _BC_labeler()
        self.imgs = imgs
        self.labels = labels      

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx] 
        label = self.labels[idx]
        image_size = self.image_size # Get image_size

        # Pre-process task
        if self.modality == 'CT':
            img = util.CT_preprocess(img, image_size)*2 - 1 # Data value range : -1.0 ~ 1.0
            albu_dic = self.transform(image=img) # Apply augmentation to data
        elif self.modality == 'X-ray':
            img = util.Xray_preprocess_minmax(img, image_size)*2 - 1 # Data value range : -1.0 ~ 1.0
            albu_dic = self.transform(image=img)
            '''
            img = Xray_preprocess_percentile(img, image_size)*2 - 1 # Data value range : -1.0 ~ 1.0
            albu_dic = self.transform(image=img)
            '''
        elif self.modality == 'Endo':
            img = util.Endo_preprocess(img, image_size)*2-1
            albu_dic = self.transform(image=img)

        data_dic= {'image' : albu_dic['image'], 'label' : label} # Make dictionary which has one data to image key and one label to label key
        return data_dic
    

# get loader from Dataset as batch size
# img_dir, label_dir, preprocess_type, transform, batch_size, workers
def get_loader(args):
    dataset = Dataset(args.data_dir, args.img_size, args.modality, args.augmentation)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            shuffle=args.shuffle_set, num_workers=args.workers,
                            drop_last=args.drop_last)
    return dataloader


