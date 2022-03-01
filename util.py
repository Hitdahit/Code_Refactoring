import os
import cv2
import numpy as np
import pydicom



'''
preprocessor를 util로 보낼 것인가?
'''
def Endo_preprocess(data, image_size):
    rgb_image = cv2.imread(data)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    if rgb_image.shape == (1920, 1080):
        rgb_image = rgb_image[680:-1, :]
    elif rgb_image.shape == (640, 480):
        rgb_image =  rgb_image[180:-1, :]
    rgb_image = cv2.resize(rgb_image, dsize=image_size)

    rgb_image = (rgb_image - np.min(rgb_image))/(np.max(rgb_image)-np.min(rgb_image))

    return rgb_image

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