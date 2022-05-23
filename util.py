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
def CT_preprocess(data, image_size, window_width=None, window_level=None): # (input data, image size which you want to resize, window width which you want, window level which you want)
    dicom_image = pydicom.read_file(data) # Read dicom file
    pixel_array_image = dicom_image.pixel_array.astype(np.float32) # Change dicom image to array

    # If dicom image has 3 channel, change it to 1 channel
    if len(pixel_array_image.shape) == 3:
        pixel_array_image = pixel_array_image[:,:,0]
    
    pixel_array_image = pixel_array_image.squeeze() # Delete 1 dimension in array -> make (512,512)

    # If image_size is not 512, change it to 512
    if pixel_array_image.shape != (image_size, image_size):
        pixel_array_image = cv2.resize(pixel_array_image, (image_size, image_size))

    # CT image has Rescale Intercept and Rescale Slope. It is mandantory pre-process task.
    intercept = dicom_image.RescaleIntercept
    slope = dicom_image.RescaleSlope

    pixel_array_image = pixel_array_image * slope + intercept

    # If you give a specific value in window_width and window_level, it will be windowed by value of those.
    if not window_width == None and window_level == None:
        image_min = window_level - (window_width / 2.0)
        image_max = window_level + (window_width / 2.0)

        # If image pixel is over max value or under min value, threshold to max and min
        pixel_array_image[np.where(pixel_array_image < image_min)] = image_min
        pixel_array_image[np.where(pixel_array_image > image_max)] = image_max

        pixel_array_image = (pixel_array_image - image_min) / (image_max - image_min)
    else:
        image_min = -1024.0
        image_max = 3071.0

        # If image pixel is over max value or under min value, threshold to max and min
        pixel_array_image[np.where(pixel_array_image < image_min)] = image_min
        pixel_array_image[np.where(pixel_array_image > image_max)] = image_max

        pixel_array_image = (pixel_array_image - image_min) / (image_max - image_min)

    # If dicom image has MONOCHROME1, min value of image is white pixel, while max value of image is black pixel
    # In other words, it is conversion version of image that we know conventionally. So it has to be converted
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

    pixel_array_image = (pixel_array_image - np.min(pixel_array_image)) / (np.max(pixel_array_image) - np.min(pixel_array_image))

    # If dicom image has MONOCHROME1, min value of image is white pixel, while max value of image is black pixel
    # In other words, it is conversion version of image that we know conventionally. So it has to be converted
    if dicom_image.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array_image = 1.0 - pixel_array_image

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

    # pixel value is divided by Percentile 99% and normalized from 0 to 1
    pixel_array_image = (pixel_array_image - np.min(pixel_array_image)) / (np.max(pixel_array_image) - np.min(pixel_array_image))
    pixel_array_image /= np.percentile(pixel_array_image, 99)

    # If dicom image has MONOCHROME1, min value of image is white pixel, while max value of image is black pixel
    # In other words, it is conversion version of image that we know conventionally. So it has to be converted
    if dicom_image.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array_image = 1.0 - pixel_array_image

    return pixel_array_image # output : 0.0 ~ 1.0