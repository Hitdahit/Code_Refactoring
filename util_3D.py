from code import InteractiveInterpreter
import os
import cv2
import numpy as np
import nibabel as nib


'''
3D preprocessor도 util로 따로
'''
# CT pre-processing
def CT_preprocess(data, image_size, window_width=None, window_level=None): # (input data, image size which you want to resize, window width which you want, window level which you want)
    nifti_file = nib.load(data) # Read nifti file
    nifti_np_array = nifti_file.get_fdata(dtype=np.float32) # shape(512,512,?)

    if nifti_np_array == 0:
        print('Data is empty. Data needs to be non-empty')

    intercept = nifti_file.dataobj.inter
    slope = nifti_file.dataobj.slope

    nifti_np_array = nifti_np_array * slope + intercept


    return ddong
