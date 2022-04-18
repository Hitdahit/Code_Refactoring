import os
import cv2
import numpy as np
import nibabel as nib
import SimpleITK as sitk


## Read nifti file
def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(image, eps=0):
    image = (image - image.min())/(image.max()-image.min()+eps)
    image= image*255
    return image.astype(np.uint8)


## Resample voxel spacing
def MR_resampling(data, voxel_size):
    images = sitk.ReadImage(data)
    ori_spacing = images.GetSpacing()
    ori_size = images.GetSize()
    out_size = [int(ori_size[0] * (ori_spacing[0] / voxel_size)),
                int(ori_size[1] * (ori_spacing[1] / voxel_size)),
                int(ori_size[2] * (ori_spacing[2] / voxel_size)),]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing([voxel_size, voxel_size, voxel_size])
    resample.SetSize(out_size)
    resample.SetOutputDirection(images.GetDirection())
    resample.SetOutputOrigin(images.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(images.GetPixelIDValue())
    resampled_image = resample.Execute(images)

    return resampled_image

        
if __name__ == '__main__':      
    
    train_img_dir = 'directory of train image'
    train_msk_dir = 'directory of train mask'

    train_img_name = os.listdir('directory of train image')
    train_msk_name = os.listdir('directory of train mask')
    
    voxel_size = 1  # parser

    for i in train_img_name:
        resampled_img = MR_resampling(train_img_dir + i, voxel_size)
        images = sitk.GetArrayFromImage(resampled_img)  # shape : (z, y, x)
        # print(images.shape)
        
        split_name = i.split('.')
        new_name = split_name[0]
        
        
        for index in range(images.shape[0]):
            image = normalize(images[index])
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR) 
            
            filename = new_name + '_%02d.png'%(index+1)
            cv2.imwrite(train_img_dir + filename, image)
            
    for i in train_msk_name:
        images = sitk.GetArrayFromImage(sitk.ReadImage(train_msk_dir + i))  # shape : (z, y, x)
        # print(images.shape)
        
        split_name = i.split('.')
        new_name = split_name[0]
        
        for index in range(images.shape[0]):
            image = normalize(images[index])
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST) 
            
            filename = new_name + '_%02d.png'%(index+1)
            cv2.imwrite(train_msk_dir + filename, image)
    