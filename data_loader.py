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
    def __init__(self, img_dir, label_dir, image_size, preprocess_type, transform):
        self.data = glob.glob(os.path.join(img_dir, '*')) # Get data path list
        self.label = glob.glob(os.path.join(label_dir, '*')) # Get label path list
        self.image_size = image_size # Get image_size
        self.preprocess_type = preprocess_type # Get preprocess_type. For example, CT or X-ray_minmax or X-ray_percentile

        self.data_dictionary = {} # Create dictionary for image and label
        self.data_dictionary['image'] = self.data # Make image key in dictionary and append data list
        self.data_dictionary['label'] = self.label # Make label key in dictionary and append label list

        self.transform = transform # Get augmentation

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data_dictionary[idx]['image'] # Get one data item in data_dictionary
        image_size = self.image_size # Get image_size

        # Pre-process task
        if self.preprocess_type == 'CT':
            data = CT_preprocess(data, image_size)*2 - 1 # Data value range : -1.0 ~ 1.0
            data = self.transform(image=data) # Apply augmentation to data
        elif self.preprocess_type == 'X-ray_minmax':
            data = Xray_preprocess_minmax(data, image_size)*2 - 1 # Data value range : -1.0 ~ 1.0
            data = self.transform(image=data)
        elif self.preprocess_type == 'X-ray_percentile':
            data = Xray_preprocess_percentile(data, image_size)*2 - 1 # Data value range : -1.0 ~ 1.0
            data = self.transform(image=data)

        label = self.data_dictionary[idx]['label'] # Get one label item in data_dictionary

        data_dictionary = {'image' : data, 'label' : label} # Make dictionary which has one data to image key and one label to label key
        return data_dictionary
    

# get loader from Dataset as batch size
def get_loader(img_dir, label_dir, preprocess_type, transform, batch_size, workers):
    dataset = Dataset(img_dir, label_dir, preprocess_type, transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers,
                            drop_last=True)
    return dataloader


# CT pre-processing
def CT_preprocess(data, image_size):
    dicom_image = pydicom.read_file(data) # Read dicom file
    pixel_array_image = dicom_image.pixel_array.astype(np.float32) # Change dicom image to array

    # If dicom image has 3 channel, change it to 1 channel
    if len(pixel_array_image.shape) == 3:
        pixel_array_image = pixel_array_image[:,:,0]
    
    pixel_array_image = pixel_array_image.squeeze() # Delete 1 dimension in array -> make (512,512)

    # If image_size is not 512, change it to 512
    if pixel_array_image.shape != (image_size, image_size):
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size))

    # CT image has RescaleIntercept and Rescale Slope. It is mandantory pre-process task.
    intercept = dicom_image.RescaleIntercept
    slope = dicom_image.RescaleSlope

    if ('RescaleSlope' in dicom_image) and ('RescaleIntercept' in dicom_image):
        pixel_array_image = pixel_array_image * slope + intercept

    pixel_array_image = (pixel_array_image - np.min(pixel_array_image)) / (2.0**dicom_image.BitsStored-1.0) # CT image has 8, 12, 16bits. It must be normalized.

    image_min = np.min(pixel_array_image)
    image_max = np.max(pixel_array_image)

    # If image pixel is over max value or under min value, threshold to max and min
    pixel_array_image[np.where(pixel_array_image < image_min)] = image_min
    pixel_array_image[np.where(pixel_array_image > image_max)] = image_max

    pixel_array_image = (pixel_array_image-image_min) / (image_max-image_min) # Normalize from 0 to 1

    # If dicom image has MONOCHROME1, image is converted.
    if dicom_image.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array_image = 1.0 - pixel_array_image

    return pixel_array_image # output : 0.0 ~ 1.0


# X-ray pre-process by using min-max scaling
def Xray_preprocess_minmax(data, image_size):
    dicom_image = pydicom.read_file(data) # Read dicom file
    pixel_array_image = dicom_image.pixel_array.astype(np.float32) # Change dicom image to array

    # If dicom image has 3 channel, change it to 1 channel
    if len(pixel_array_image.shape) == 3:
        pixel_array_image = pixel_array_image[:,:,0]
    
    pixel_array_image = pixel_array_image.squeeze() # Delete 1 dimension in array -> make (512,512)

    # If image_size is not 512, change it to 512
    if pixel_array_image.shape != (image_size, image_size):
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size))

    pixel_array_image = (pixel_array_image - np.min(pixel_array_image)) / (2.0**dicom_image.BitsStored-1.0) # X-ray image has 8, 12, 16bits. It must be normalized.

    image_min = np.min(pixel_array_image)
    image_max = np.max(pixel_array_image)

    # If image pixel is over max value or under min value, threshold to max and min
    pixel_array_image[np.where(pixel_array_image < image_min)] = image_min
    pixel_array_image[np.where(pixel_array_image > image_max)] = image_max

    pixel_array_image = (pixel_array_image-image_min) / (image_max-image_min) # Normalize from 0 to 1

    return pixel_array_image # output : 0.0 ~ 1.0


# X-ray pre-process by using percentile
def Xray_preprocess_percentile(data, image_size):
    dicom_image = pydicom.read_file(data) # Read dicom file
    pixel_array_image = dicom_image.pixel_array.astype(np.float32) # Change dicom image to array

    # If dicom image has 3 channel, change it to 1 channel
    if len(pixel_array_image.shape) == 3:
        pixel_array_image = pixel_array_image[:,:,0]
    
    pixel_array_image = pixel_array_image.squeeze() # Delete 1 dimension in array -> make (512,512)

    # If image_size is not 512, change it to 512
    if pixel_array_image.shape != (image_size, image_size):
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size))

    pixel_array_image = (pixel_array_image - np.min(pixel_array_image)) / (2.0**dicom_image.BitsStored-1.0) # X-ray image has 8, 12, 16bits. It must be normalized.

    pixel_array_image -= np.min(pixel_array_image) # Start pixel value from 0
    pixel_array_image /= np.percentile(pixel_array_image, 99) # Normalize from 0 to 1

    # If image pixel is over 1.0 or under 0.0, threshold to 1.0 and 0.0
    pixel_array_image[np.where(pixel_array_image < 0.0)] = 0.0
    pixel_array_image[np.where(pixel_array_image > 1.0)] = 1.0

    return pixel_array_image # output : 0.0 ~ 1.0