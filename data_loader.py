import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import glob
import pydicom
import numpy as np

from train import Albumentations

# CT pre-processing
def CT_preprocess(data):
    dicom_image = pydicom.read_file(data)

    pixel_array_image = dicom_image.pixel_array.astype(np.float32)

    intercept = dicom_image.RescaleIntercept
    slope     = dicom_image.RescaleSlope

    if ('RescaleSlope' in dicom_image) and ('RescaleIntercept' in dicom_image):
        pixel_array_image = pixel_array_image * slope + intercept

    image_min = -1024.0
    image_max = 3071.0

    pixel_array_image[np.where(pixel_array_image < image_min)] = image_min
    pixel_array_image[np.where(pixel_array_image > image_max)] = image_max

    pixel_array_image = (pixel_array_image-image_min) / (image_max-image_min)

    if dicom_image.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array_image = 1.0 - pixel_array_image

    return pixel_array_image # output : 0.0 ~ 1.0


# X-ray pre-process
def Xray_preprocess(data):
    dicom_image = pydicom.read_file(data)

    pixel_array_image = dicom_image.pixel_array.astype(np.float32).squeeze()
    pixel_array_image = pixel_array_image / (2.0 ** dicom_image.BitsStored)
    if pixel_array_image.shape != (512, 512):
        pixel_array_image = cv2.resize(pixel_array_image, (512, 512))
    if len(pixel_array_image.shape) == 3:
        pixel_array_image = pixel_array_image[:,:,0]

    pixel_array_image -= np.min(pixel_array_image)
    pixel_array_image /= np.percentile(pixel_array_image, 99)
    pixel_array_image[pixel_array_image > 1] = 1

    return pixel_array_image # output : 0.0 ~ 1.0


# Make Dataset
class Dataset(Dataset):
    def __init__(self, img_dir, label_dir, data_type):
        self.data = glob.glob(os.path.join(img_dir, '*'))
        self.label = glob.glob(os.path.join(label_dir, '*'))
        self.data_type = data_type

        self.data_dictionary = {}
        self.data_dictionary['image'] = self.data
        self.data_dictionary['label'] = self.label

        self.transform = Albumentations()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.data_type == 'CT':
            data = CT_preprocess(self.data_dictionary[idx]['image'])*2 - 1 # value range : -1.0 ~ 1.0
            data = self.transform(data)
        elif self.data_type == 'X-ray':
            data = Xray_preprocess(self.data_dictionary[idx]['image'])*2 - 1 # value range : -1.0 ~ 1.0
            data = self.transform(data)

        label = self.data_dictionary[idx]['label']

        data_dictionary = {'image' : data, 'label' : label}
        return data_dictionary
    

# get loader from Dataset as batch size
def get_loader(img_dir, label_dir, data_type, batch_size, workers):
    dataset = Dataset(img_dir, label_dir, data_type)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers,
                            drop_last=True)
    return dataloader