import os
import cv2
import argparse
import numpy as np
import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='', type=str)
parser.add_argument('--save_dir', default='', type=str)
parser.add_argument('--view', default='coronal', type=str)
   
if __name__ == '__main__':      
    args = parser.parse_args()
    
    train_img_dir = args.train_dir + '/image'
    train_msk_dir = args.train_dir + '/mask'

    train_img_name = os.listdir(train_img_dir)
    train_msk_name = os.listdir(train_msk_dir)
    
    ## check origin of input image
    check_image = sitk.ReadImage(train_img_dir + train_img_name[0])
    origin = check_image.GetOrigin()
    #print('Origin of image: ', origin)

    for i in train_img_name:
        images = sitk.ReadImage(train_img_dir + i)
        images.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        images.SetOrigin(origin)
        
        images_array = sitk.GetArrayFromImage(images)  # shape : (z, y, x)
        # print(images.shape)
        
        split_name = i.split('.')
        new_name = split_name[0]
        
        for index in range(images_array.shape[0]):
            
            if args.view == 'coronal':
                image = images_array[index, :, :]  
            elif args.view == 'axial':
                image = images_array[:, index, :]  
            elif args.view == 'sagittal':
                image = images_array[:, :, index]  
                        
            filename = new_name + '_%02d.npy'%(index+1)
            np.save(args.save_dir + filename, image)
            
    '''       
    If you use segmentation masks,

    for i in train_msk_name:
        masks = sitk.ReadImage(train_msk_dir + i)
        masks.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        masks.SetOrigin(origin)
        
        masks_array = sitk.GetArrayFromImage(masks)  # shape : (z, y, x)
        # print(masks.shape)
        
        split_name = i.split('.')
        new_name = split_name[0]
        
        for index in range(masks_array.shape[0]):
        
            if args.view == 'coronal':
                image = images_array[index, :, :]  
            elif args.view == 'axial':
                image = images_array[:, index, :]  
            elif args.view == 'sagittal':
                image = images_array[:, :, index] 
                        
            filename = new_name + '_%02d.npy'%(index+1)
            np.save(args.save_dir + filename, mask)
    '''
  