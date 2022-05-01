import os
import cv2
import numpy as np
import nibabel as nib

from scipy.ndimage import zoom
'''
3D preprocessor도 util로 따로
'''
# CT pre-processing
def CT_preprocess(data, image_size, window_width=None, window_level=None, normalization='mean'): # (input data, image size which you want to resize, window width which you want, window level which you want)
    nifti_file = nib.load(data) # Read nifti file
    nifti_np_array = nifti_file.get_fdata(dtype=np.float32)

    # Alarm that data is empty.
    if nifti_np_array == 0:
        print('Data is empty. Data needs to be non-empty')

    # Resize data e.g. (256,256,128) -> (512,512,128)
    if nifti_np_array[:,:,0].shape != (image_size, image_size):
        nifti_np_array = resize(nifti_np_array, image_size)

    # CT image has Rescale Intercept and Rescale Slope. It is mandantory pre-process task.
    intercept = nifti_file.dataobj.inter
    slope = nifti_file.dataobj.slope

    nifti_np_array = nifti_np_array * slope + intercept
    
    # If you give a specific value in window_width and window_level, it will be windowed by value of those.
    nifti_np_array = windowing(nifti_np_array, window_width, window_level)
    
    # Normalize intensity. There is two options, one is mean/std normalization and the other is min-max normalization.
    nifti_np_array = normalize_intensity(nifti_np_array, normalization)

    return nifti_np_array


def patch():
    '''
    will be updated...
    '''


def resize(image, image_size):
    scale1 = image_size / image[:,0,0].shape
    scale2 = image_size / image[0,:,0].shape
    scale3 = 1.0

    scale_list = [scale1, scale2, scale3]
    image = zoom(image, scale_list, order=0)
    return image


def windowing(image, window_width=None, window_level=None):
    if not window_width == None and window_level == None:
        image_min = window_level - (window_width / 2.0)
        image_max = window_level + (window_width / 2.0)

        # If image pixel is over max value or under min value, threshold to max and min
        image = np.clip(image, image_min, image_max)
    else:
        image_min = -1024.0
        image_max = 3071.0

        # If image pixel is over max value or under min value, threshold to max and min
        image = np.clip(image, image_min, image_max)
    return image


def normalize_intensity(image, normalization='mean'):
    if normalization == 'mean':
        mean, std = np.mean(image), np.std(image)
        image = (image - mean) / std
    elif normalization == 'max':
        min, max = np.min(image), np.max(image)
        image = (image - min) / (max - min)
    return image