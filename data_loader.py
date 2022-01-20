from curses import window
import torch
import torchvision

import os
import glob
import pydicom
import numpy as np
import albumentations
import SimpleITK as sitk
import cv2

from PIL import Image
from torch.utils.data import Dataset

# Make Dataset
class Dataset(Dataset):
    def __init__(self, img_dir, label_dir, data_type):
        self.data      = glob.glob(os.path.join(img_dir, '*'))
        self.label     = glob.glob(os.path.join(label_dir, '*'))
        self.data_type = data_type

        self.data_dictionary          = {}
        self.data_dictionary['image'] = self.data
        self.data_dictionary['label'] = self.label

        self.transform = Albumentations()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.data_type == 'CT':
            data = self.transform(CT_preprocess(self.data_dictionary['image'][idx]))
        elif self.data_type == 'X-ray':
            data = self.transform(Xray_preprocess(self.data_dictionary['image'][idx]))
        else:
            data = self.transform(Image.open(self.data_dictionary['image'][idx]))

        label = self.data_dictionary['label'][idx]

        data_dictionary = {}
        data_dictionary = self.data_dictionary
        return data_dictionary
    

# get loader from Dataset as batch size
def get_loader(img_dir, label_dir, data_type, batch_size, workers):
    dataset    = Dataset(img_dir, label_dir, data_type)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=workers, drop_last=True)
    return dataloader


# Data transform
def Albumentations():
    Albumentations_transform = albumentations.Compose([
        # albumentations.Resize(256, 256), 
        # albumentations.RandomCrop(224, 224),
        # albumentations.HorizontalFlip(),
        albumentations.pytorch.transforms.ToTensor()
    ])
    return Albumentations_transform


# CT pre-processing
def CT_preprocess(data):
    dicom_image = pydicom.read_file(data)

    pixel_array_image = dicom_image.pixel_array.astype(np.float32)
    pixel_array_image = (pixel_array_image / (2**dicom_image.BitsStored))

    intercept = dicom_image.RescaleIntercept
    slope     = dicom_image.RescaleSlope

    if ('RescaleSlope' in dicom_image) and ('RescaleIntercept' in dicom_image):
        pixel_array_image = pixel_array_image * slope + intercept

    if('WindowCenter' in dicom_image):
        if(type(dicom_image.Windowcenter) == pydicom.multival.MultiValue):
            window_center = float(dicom_image.WindowCenter[0])
            window_width  = float(dicom_image.WindowWidth[0])
            image_min     = window_center - window_width // 2
            image_max     = window_center + window_width // 2
        else:
            window_center = float(dicom_image.WindowCenter)
            window_width  = float(dicom_image.WindowWidth)
            image_min     = window_center - window_width // 2
            image_max     = window_center + window_width // 2
    else:
        image_min = np.min(pixel_array_image)
        image_max = np.max(pixel_array_image)

    pixel_array_image[np.where(pixel_array_image < image_min)] = image_min
    pixel_array_image[np.where(pixel_array_image > image_max)] = image_max

    pixel_array_image = (pixel_array_image-image_min) / (image_max-image_min)

    return pixel_array_image # output : 0.0 ~ 1.0


# X-ray pre-process
def Xray_preprocess(data):
    dicom_image = pydicom.read_file(data)

    image = dicom_image.pixel_array.astype(np.float32).squeeze()
    if image.shape != (512, 512):
        img = cv2.resize(image, (512, 512))
    if len(image.shape) == 3:
        image = image[:,:,0]

    np_image = img.astype(np.float32)
    np_image -= np.min(np_image)
    np_image /= np.percentile(np_image, 99)
    np_image[np_image > 1] = 1

    return np_image # output : 0.0 ~ 1.0
