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
import utils.image_preprocess as image_preprocess

# Make Dataset
class Dataset(Dataset):

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

    def __init__(self, img_dir, mode, image_size, modality, transform, window_width, window_level, photometricInterpretation, normalize_range, percentage):
        '''
        data's root dir
        '''
        self.img_dir = os.path.join(img_dir, mode)
        self.image_size = image_size # Get image_size
        self.modality = modality        
        self.transform = transform # Get augmentation

        self.window_width = window_width
        self.window_width = window_level
        self.photometricInterpretation = photometricInterpretation
        self.normalize_range = normalize_range
        self.percentage = percentage

        self.classes =sorted( os.listdir(self.img_dir))
        
        imgs, labels = self._BC_labeler()
        self.imgs = imgs
        self.labels = labels      

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx] 
        label = self.labels[idx]
        image_size = self.image_size # Get image_size
        photometricInterpretation = self.photometricInterpretation
        normalize_range = self.normalize_range

        # Pre-process task
        if self.modality == 'CT':
            window_width = self.window_width
            window_level = self.window_level

            img = image_preprocess.CT_preprocess(img, image_size, window_width, window_level, photometricInterpretation, normalize_range)
            albu_dic = self.transform(image=img) # Apply augmentation to data
        elif self.modality == 'X-ray':
            img = image_preprocess.Xray_preprocess_minmax(img, image_size, photometricInterpretation, normalize_range)
            albu_dic = self.transform(image=img)
            '''
            percentage = self.percentage
            img = Xray_preprocess_percentile(img, image_size, photometricInterpretation, normalize_range, percentage)
            albu_dic = self.transform(image=img)
            '''
        elif self.modality == 'Endo':
            img = image_preprocess.Endo_preprocess(img, image_size, photometricInterpretation, normalize_range)
            albu_dic = self.transform(image=img)
            
        elif self.modality == 'MR':
            img = image_preprocess.MR_preprocess(img, image_size)
            albu_dic = self.transform(image=img)

        data_dic= {'image' : albu_dic['image'], 'label' : label} # Make dictionary which has one data to image key and one label to label key
        return data_dic
    

# get loader from Dataset as batch size
# img_dir, label_dir, preprocess_type, transform, batch_size, workers
def get_loader(args, mode='train'):
    dataset = Dataset(args.data_dir, mode, args.img_size, args.modality, args.augmentation,
                      args.window_width, args.window_level, args.photometricInterpretation, args.normalize_range, args.percentage)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            shuffle=args.shuffle, num_workers=args.num_worker,
                            drop_last=args.drop_last)
    return dataloader


