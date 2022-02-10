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
    def __init__(self, img_dir, label_dir, image_size, data_type, transform):
        self.data = glob.glob(os.path.join(img_dir, '*'))
        self.label = glob.glob(os.path.join(label_dir, '*'))
        self.image_size = image_size
        self.data_type = data_type

        self.data_dictionary = {}
        self.data_dictionary['image'] = self.data
        self.data_dictionary['label'] = self.label

        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data_dictionary[idx]['image']
        image_size = self.image_size

        if self.data_type == 'CT':
            data = CT_preprocess(data, image_size)*2 - 1 # value range : -1.0 ~ 1.0
            data = self.transform(image=data)
        elif self.data_type == 'X-ray_minmax':
            data = Xray_preprocess_minmax(data, image_size)*2 - 1 # value range : -1.0 ~ 1.0
            data = self.transform(image=data)
        elif self.data_type == 'X-ray_percentile':
            data = Xray_preprocess_percentile(data, image_size)*2 - 1 # value range : -1.0 ~ 1.0
            data = self.transform(image=data)

        label = self.data_dictionary[idx]['label']

        data_dictionary = {'image' : data, 'label' : label}
        return data_dictionary
    

# get loader from Dataset as batch size
def get_loader(img_dir, label_dir, data_type, transform, batch_size, workers):
    dataset = Dataset(img_dir, label_dir, data_type, transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers,
                            drop_last=True)
    return dataloader


# CT pre-processing
def CT_preprocess(data, image_size):
    dicom_image = pydicom.read_file(data)
    pixel_array_image = dicom_image.pixel_array.astype(np.float32)

    if len(pixel_array_image.shape) == 3:
        pixel_array_image = pixel_array_image[:,:,0]
    
    pixel_array_image = pixel_array_image.squeeze()

    if pixel_array_image.shape != (image_size, image_size):
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size))

    intercept = dicom_image.RescaleIntercept
    slope = dicom_image.RescaleSlope

    if ('RescaleSlope' in dicom_image) and ('RescaleIntercept' in dicom_image):
        pixel_array_image = pixel_array_image * slope + intercept

    pixel_array_image = (pixel_array_image - np.min(pixel_array_image)) / (2.0**dicom_image.BitsStored-1.0)

    image_min = 0.0
    image_max = 1.0

    pixel_array_image[np.where(pixel_array_image < image_min)] = image_min
    pixel_array_image[np.where(pixel_array_image > image_max)] = image_max

    pixel_array_image = (pixel_array_image-image_min) / (image_max-image_min)

    if dicom_image.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array_image = 1.0 - pixel_array_image

    return pixel_array_image # output : 0.0 ~ 1.0


# X-ray pre-process by using min-max scaling
def Xray_preprocess_minmax(data, image_size):
    dicom_image = pydicom.read_file(data)
    pixel_array_image = dicom_image.pixel_array.astype(np.float32)

    if len(pixel_array_image.shape) == 3:
        pixel_array_image = pixel_array_image[:,:,0]
    
    pixel_array_image = pixel_array_image.squeeze()

    if pixel_array_image.shape != (image_size, image_size):
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size))

    pixel_array_image = pixel_array_image / (2.0**dicom_image.BitsStored-1.0)

    image_min = np.min(pixel_array_image)
    image_max = np.max(pixel_array_image)

    pixel_array_image = (pixel_array_image-image_min) / (image_max-image_min)
    pixel_array_image[np.where(pixel_array_image < 0.0)] = 0.0
    pixel_array_image[np.where(pixel_array_image > 1.0)] = 1.0

    return pixel_array_image # output : 0.0 ~ 1.0


# X-ray pre-process by using percentile
def Xray_preprocess_percentile(data, image_size):
    dicom_image = pydicom.read_file(data)
    pixel_array_image = dicom_image.pixel_array.astype(np.float32)

    if len(pixel_array_image.shape) == 3:
        pixel_array_image = pixel_array_image[:,:,0]
    
    pixel_array_image = pixel_array_image.squeeze()

    if pixel_array_image.shape != (image_size, image_size):
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size))

    pixel_array_image = pixel_array_image / (2.0**dicom_image.BitsStored-1.0)

    pixel_array_image -= np.min(pixel_array_image)
    pixel_array_image /= np.percentile(pixel_array_image, 99)
    pixel_array_image[np.where(pixel_array_image < 0.0)] = 0.0
    pixel_array_image[np.where(pixel_array_image > 1.0)] = 1.0

    return pixel_array_image # output : 0.0 ~ 1.0