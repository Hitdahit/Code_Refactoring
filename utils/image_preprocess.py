import os
import cv2
import numpy as np
import pydicom



'''
preprocessor를 util로 보낼 것인가?
'''
def Endo_preprocess(data, image_size):
    #rgb_image = cv2.imread(data)
    img = pydicom.dcmread(data)
    rgb_image = img.pixel_array
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    if rgb_image.shape == (1920, 1080):
        rgb_image = rgb_image[680:-1, :]
    elif rgb_image.shape == (640, 480):
        rgb_image =  rgb_image[180:-1, :]
    rgb_image = cv2.resize(rgb_image, dsize=image_size)

    rgb_image = (rgb_image - np.min(rgb_image))/(np.max(rgb_image)-np.min(rgb_image))
    rgb_image = np.array(rgb_image, dtype=np.float32)
    return rgb_image

# CT pre-processing
def CT_preprocess(data, image_size, window_width=None, window_level=None, photometricInterpretation='MONOCHROME2', normalize_range='1'): # (input data, image size which you want to resize, window width which you want, window level which you want).
    dicom_image = pydicom.read_file(data) # Read dicom file
    pixel_array_image = dicom_image.pixel_array.astype(np.float32) # Change dicom image to array.

    # If dicom image has 3 channel, change it to 1 channel.
    if len(pixel_array_image.shape) == 3:
        pixel_array_image = pixel_array_image[:,:,0]
    
    pixel_array_image = pixel_array_image.squeeze() # Delete 1 dimension in array -> make (512,512)

    # If image_size is not what you want, change it.
    if pixel_array_image.shape[0] < image_size and pixel_array_image.shape[1] < image_size:
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size), cv2.INTER_CUBIC)
    elif pixel_array_image.shape[0] > image_size and pixel_array_image.shape[1] > image_size:
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size), cv2.INTER_AREA)

    # CT image has Rescale Intercept and Rescale Slope. It is mandantory pre-process task.
    intercept = dicom_image.RescaleIntercept
    slope = dicom_image.RescaleSlope

    pixel_array_image = pixel_array_image * slope + intercept

    # If you give a specific value in window_width and window_level, it will be windowed by value of those.
    if window_width == None and window_level == None:
        image_min = -1024.0
        image_max = 3071.0
    else:
        image_min = window_level - (window_width / 2.0)
        image_max = window_level + (window_width / 2.0)

    # If image pixel is over max value or under min value, threshold to max and min.
    pixel_array_image = np.clip(pixel_array_image, image_min, image_max)

    pixel_array_image = (pixel_array_image - image_min) / (image_max - image_min)

    # If dicom image has MONOCHROME1, min value of image is white pixel, while max value of image is black pixel.
    # In other words, it is conversion version of image that we know conventionally. So it has to be converted.
    if dicom_image.PhotometricInterpretation != photometricInterpretation:
        pixel_array_image = 1.0 - pixel_array_image
    
    # Normalization has two types, 0~1 or -1~1
    if normalize_range == '1':
        pass
    elif normalize_range == '2':
        pixel_array_image = pixel_array_image * 2 - 1

    return pixel_array_image

# X-ray pre-process by using min-max scaling
def Xray_preprocess_minmax(data, image_size, photometricInterpretation='MONOCHROME2', normalize_range='1'):
    dicom_image = pydicom.read_file(data) # Read dicom file
    pixel_array_image = dicom_image.pixel_array.astype(np.float32) # Change dicom image to array

    # If dicom image has 3 channel, change it to 1 channel
    if len(pixel_array_image.shape) == 3:
        pixel_array_image = pixel_array_image[:,:,0]
    
    pixel_array_image = pixel_array_image.squeeze() # Delete 1 dimension in array -> make (512,512)

    # If image_size is not what you want, change it.
    if pixel_array_image.shape[0] < image_size and pixel_array_image.shape[1] < image_size:
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size), cv2.INTER_CUBIC)
    elif pixel_array_image.shape[0] > image_size and pixel_array_image.shape[1] > image_size:
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size), cv2.INTER_AREA)

    pixel_array_image = (pixel_array_image - np.min(pixel_array_image)) / (np.max(pixel_array_image) - np.min(pixel_array_image))

    # If dicom image has MONOCHROME1, min value of image is white pixel, while max value of image is black pixel
    # In other words, it is conversion version of image that we know conventionally. So it has to be converted
    if dicom_image.PhotometricInterpretation != photometricInterpretation:
        pixel_array_image = 1.0 - pixel_array_image
        
    # Normalization has two types, 0~1 or -1~1
    if normalize_range == '1':
        pass
    elif normalize_range == '2':
        pixel_array_image = pixel_array_image * 2 - 1

    return pixel_array_image


# X-ray pre-process by using percentile
def Xray_preprocess_percentile(data, image_size, photometricInterpretation='MONOCHROME2', normalize_range='1'):
    dicom_image = pydicom.read_file(data) # Read dicom file
    pixel_array_image = dicom_image.pixel_array.astype(np.float32) # Change dicom image to array

    # If dicom image has 3 channel, change it to 1 channel
    if len(pixel_array_image.shape) == 3:
        pixel_array_image = pixel_array_image[:,:,0]
    
    pixel_array_image = pixel_array_image.squeeze() # Delete 1 dimension in array -> make (512,512)

    # If image_size is not what you want, change it.
    if pixel_array_image.shape[0] < image_size and pixel_array_image.shape[1] < image_size:
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size), cv2.INTER_CUBIC)
    elif pixel_array_image.shape[0] > image_size and pixel_array_image.shape[1] > image_size:
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size), cv2.INTER_AREA)

    # pixel value is divided by Percentile 99% and normalized from 0 to 1
    ninetynine = np.percentile(pixel_array_image, 99)
    one = np.percentile(pixel_array_image, 1)
    pixel_array_image = np.clip(pixel_array_image, one, ninetynine)
    
    pixel_array_image = (pixel_array_image - np.min(pixel_array_image)) / (np.max(pixel_array_image) - np.min(pixel_array_image))

    # If dicom image has MONOCHROME1, min value of image is white pixel, while max value of image is black pixel
    # In other words, it is conversion version of image that we know conventionally. So it has to be converted
    if dicom_image.PhotometricInterpretation != photometricInterpretation:
        pixel_array_image = 1.0 - pixel_array_image
        
    # Normalization has two types, 0~1 or -1~1
    if normalize_range == '1':
        pass
    elif normalize_range == '2':
        pixel_array_image = pixel_array_image * 2 - 1

    return pixel_array_image
    
'''
Metirc
BC: binary class, ML: multi label, MC: multi class
'''
def BC_metric(y, yhat, thresh=0.5):
    cor = 0
    yhat = torch.softmax(yhat, dim=1)
    yhat[yhat>=thresh] = 1
    yhat[yhat<thresh] = 0
    
    y = torch.argmax(y, dim=1)
    yhat = torch.argmax(yhat, dim=1)
    for i, j in zip(y, yhat):
        if i==j:
            cor = cor+1
            
    acc = cor/len(yhat)
    
    return acc

def ML_metric(y_true, y_pred):
    '''
    일반적인 accuracy (example_based_accuracy)
    전체 predicted와 actual label 중에 맞은 predicted label 비율
    
    if y_true = np.array([[0,1,1], [1,1,1]])
       y_pred = np.array([[1,0,1], [0,0,0]])
    numerator = [1,0]
    denominator = [3,3]
    instance_accuracy = [0.333333, 0]
    np.sum(instance_accuracy) : 0.33333
    '''

    # compute true positive using the logical AND operator
    numerator = np.sum(np.logical_and(y_true, y_pred), axis = 1) 

    # compute true_positive + false negatives + false positive using the logical OR operator
    denominator = np.sum(np.logical_or(y_true, y_pred), axis = 1)
    instance_accuracy = numerator/denominator

    return np.sum(instance_accuracy) # accuracy 계산 하려면 data 갯수를 나눠줘야 됨

def MC_metric(y, yhat):

    acc_targets = []
    acc_outputs = []

    y_temp = y
    for t in y_temp.view(-1,1).cpu():
        acc_targets.append(t.item()) 

    _, yhat_temp = torch.max(yhat, 1)
    for u in yhat_temp.view(-1,1).cpu():
        acc_outputs.append(u.item())

    cor = 0
    for i in range(len(acc_targets)):
        if acc_outputs[i] == acc_targets[i]:
            cor += 1

    acc = cor/len(acc_outputs)

    return acc