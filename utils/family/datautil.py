import os
import json
from re import I
import pandas as pd
import numpy as np

import cv2

'''
Labeler
    #label from directory?
    #label from path?
    #label from json / txt / excel / etc... ?
    #
    #label type
    #one-hot / ordinal / regression
    # per-patient / per-image label (CT, MR)
    
    #MC label / ML label / BC label
'''
class Source():
    '''
    Source contains file names and labels.
    '''
    def __init__(self, task_type, label_type, label_name, annotation_file=None):
        '''
        task_type:  (str) BC, MC, ML 
                    !!! BC and MC will work without annotation_file

        label_type: (str) one-hot(float), ordinal(int)

        label_name: (list: str) ['label1', 'label2', ...]
                    ex) ['Normal', 'Disease']

        annotation_file: (str) label_from_path, label_from_json, label_from_df
        '''
        self.task_type = task_type
        self.label_type = label_type
        
        self.label_name = label_name
        self.annotation = annotation_file
        
        self.n_classes = len(self.label_name)
        self.label = None

        if self.label_type =='one-hot':
            self.label = np.eye(self.n_classes)    

        elif self.label_type == 'ordinal':
            self.label = np.arange(self.n_classes)


    def get_source(self, mode, data_root):
        if self.annotation is None:
            if 'ML' in self.task_type:
                raise Exception('Multi Label classification requires annotation_file')
            
            
            try:
                '''
                BC, MC annotation file 없는 경우
                '''
                imgs = [os.listdir(os.path.join(data_root, mode, i)) for i in self.label_name]
                imgs = [os.path.join(data_root, mode, self.classes[idx], j) for idx, i in enumerate(imgs) for j in i]
                
                labels =[self.label[self._label_from_path(i)] for i in imgs]
                
                labels = np.array(labels, dtype=np.float32) if 'one-hot' in self.label_type else np.array(labels, dtype=np.int64)
                                
            except:
                raise Exception('your img data path does not contain none of the label_name.')
        
        elif 'json' in self.annotation.split('.')[-1]:
            pass
        elif 'csv' in self.annotation.split('.')[-1] or 'xlsx' in self.annotation.split('.')[-1]:
            pass

        return imgs, labels


    def execute(self, x):
        if 'from_path' in self.label_source:
            ret = self._label_from_path(x)
        elif 'from_json' in self.label_source:
            ret = self._label_from_json(x)
        elif 'from_df' in self.label_source:
            ret = self._label_from_json(x)
        return ret

    def _label_from_path(self, path):
        index = None
        for idx, i in enumerate(self.label_name):
            if i in path:
                index = idx
                break
        return index
        
            
    def _label_from_json(self):
        pass
    def _label_from_df(self):
        pass

'''
every default preprocessing is min max!
'''

# CT pre-processing
class CT_Preprocessor():
    def __init__(self, image_size, window_width=None, window_level=None, mode='default'):
        self.img_sz = image_size
        self.ww = window_width
        self.wl = window_level
        self.pixel_invert = lambda x: True if x is not 'MONOCHROME2' else False
        self.mode = mode

    def execute(self, x):
        '''
            x: input image which hasn't been processed yet.
        
        '''

        if 'default' in self.mode:
            ret = self._default_prep(x)
        else:
            '''
                if you want to use your own preprocessor, 
                make your own function and call it here.
            '''
            pass

        return ret
    
    def _default_prep(self, x):
        # Change dicom image to array and set to float
        # note that image was read by pydicom
        arr = x.pixel_array.astype(np.float32) 

        # If dicom image has 3 channel, change it to 1 channel.
        if len(arr.shape) == 3:
            arr = arr[:,:,0]

        arr = arr.squeeze() # Delete 1 dimension in array -> make (512,512)

        # If image_size is not what you want, change it.
        if arr.shape[0] < self.img_sz and arr.shape[1] < self.img_sz:
            arr = cv2.resize(arr, (self.img_sz, self.img_sz), cv2.INTER_CUBIC)

        elif arr.shape[0] > self.img_sz and arr.shape[1] > self.img_sz:
            arr = cv2.resize(arr, (self.img_sz, self.img_sz), cv2.INTER_AREA)

        # CT image has Rescale Intercept and Rescale Slope. It is mandantory pre-process task.
        # recap that x is read by pydicom!!
        intercept = x.RescaleIntercept
        slope = x.RescaleSlope

        arr = arr * slope + intercept

        # If you give a specific value in window_width and window_level, it will be windowed by value of those.
        if self.ww == None and self.wl == None:
            image_min = -1024.0
            image_max = 3071.0
        else:
            image_min = self.wl - (self.ww / 2.0)
            image_max = self.wl + (self.ww / 2.0)

        # If image pixel is over max value or under min value, threshold to max and min.
        arr = np.clip(arr, image_min, image_max)

        arr = (arr - image_min) / (image_max - image_min)

        # If dicom image has MONOCHROME1, min value of image is white pixel, while max value of image is black pixel.
        # In other words, it is conversion version of image that we know conventionally. So it has to be converted.
        arr = 1.0 - arr if self.pixel_invert(x.PhotometricInterpretation) is True else arr

        return arr

    def _custom_prep(self, x):
        pass

class XRay_Preprocessor():
    def __init__(self, image_size, normalize_range='1', mode='default'):
        self.img_sz = image_size
        
        self.normalize_range = normalize_range
        self.pixel_invert = lambda x: True if x is not 'MONOCHROME2' else False
        self.mode = mode

    def execute(self, x):
        '''
            x: input image which hasn't been processed yet.
        
        '''

        if 'default' in self.mode:
            ret = self._default_prep(x)
        elif 'percentile' in self.mode:
            ret = self._percentile_prep(x)
        else:
            '''
                if you want to use your own preprocessor, 
                make your own function and call it here.
            '''
            ret = self._custom_prep(x)
            pass

        return ret
    
    def _default_prep(self, x):
        # Change dicom image to array and set to float
        # note that image was read by pydicom
        arr = x.pixel_array.astype(np.float32) # Change dicom image to array

        # If dicom image has 3 channel, change it to 1 channel
        if len(arr.shape) == 3:
            arr = arr[:,:,0]

        arr = arr.squeeze() # Delete 1 dimension in array -> make (512,512)

        # If image_size is not what you want, change it.
        if arr.shape[0] < self.img_sz and arr.shape[1] < self.img_sz:
            arr = cv2.resize(arr, (self.img_sz, self.img_sz), cv2.INTER_CUBIC)

        elif arr.shape[0] > self.img_sz and arr.shape[1] > self.img_sz:
            arr = cv2.resize(arr, (self.img_sz, self.img_sz), cv2.INTER_AREA)

        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        # If dicom image has MONOCHROME1, min value of image is white pixel, while max value of image is black pixel
        # In other words, it is conversion version of image that we know conventionally. So it has to be converted
        arr = 1.0 - arr if self.pixel_invert(x.PhotometricInterpretation) is True else arr
        
        return arr

    def _percentile_prep(self, x, percentage=99):
        arr = x.pixel_array.astype(np.float32) # Change dicom image to array

        # If dicom image has 3 channel, change it to 1 channel
        if len(arr.shape) == 3:
            arr = arr[:,:,0]
        
        arr = arr.squeeze() # Delete 1 dimension in array -> make (512,512)

        # If image_size is not what you want, change it.
        if arr.shape[0] < self.img_sz and arr.shape[1] < self.img_sz:
            arr = cv2.resize(arr, (self.img_sz, self.img_sz), cv2.INTER_CUBIC)

        elif arr.shape[0] > self.img_sz and arr.shape[1] > self.img_sz:
            arr = cv2.resize(arr, (self.img_sz, self.img_sz), cv2.INTER_AREA)

        # pixel value is divided by Percentile 99%
        ninetynine = np.percentile(arr, percentage)
        one = np.percentile(arr, 100-percentage)
        arr = np.clip(arr, one, ninetynine)
        
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        # If dicom image has MONOCHROME1, min value of image is white pixel, while max value of image is black pixel
        # In other words, it is conversion version of image that we know conventionally. So it has to be converted
        arr = 1.0 - arr if self.pixel_invert(x.PhotometricInterpretation) is True else arr

        return arr

    def _custom_prep(self, x):
        pass

class Endo_preprocessor():
    def __init__(self, image_size, normalize_range='1', mode='default'):
        self.img_sz = image_size       
        self.mode = mode

    def execute(self, x):
        '''
            x: input image which hasn't been processed yet.
        
        '''

        if 'default' in self.mode:
            ret = self._default_prep(x)

        else:
            '''
                if you want to use your own preprocessor, 
                make your own function and call it here.
            '''
            ret = self._custom_prep(x)

        return ret
    
    def _default_prep(self, x):
        # Set to float
        # note that image was read by cv2
        arr = x.astype(np.float32)
        arr = arr.squeeze() # Delete 1 dimension in array -> make (512,512)

        # If image_size is not what you want, change it.
        resize_config = cv2.INTER_AREA if self.img_sz > arr.shape[0] else cv2.INTER_CUBIC
        arr = cv2.resize(arr, resize_config)
        
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        
        return arr

class augmenations():
    def __init__(self, ):
        '''
            if you want custom augmentations,
            write your augmentations here.

            if you use albumentations or monai,
            just jot your settings on experiments.py
        '''
        pass
    