import os
import cv2
import argparse
import numpy as np
import SimpleITK as sitk
import nibabel as nib

from scipy.ndimage import zoom

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='', type=str)
parser.add_argument('--save_dir', default='', type=str)
parser.add_argument('--view', default='coronal', type=str)
parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--window_width', default=None, type=int)
parser.add_argument('--window_level', default=None, type=int)
parser.add_argument('--normalization', default='mean', choices=['mean', 'max'], type=str)
parser.add_argument('--modality', default='MR', choices=['MR', 'CT'], type=str)
   
if __name__ == '__main__':      
    args = parser.parse_args()
    
    train_img_dir = args.train_dir + '/image'
    train_msk_dir = args.train_dir + '/mask'

    train_img_name = os.listdir(train_img_dir)
    train_msk_name = os.listdir(train_msk_dir)

    if args.modality == 'MR':
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
    elif args.modality == 'CT':
        for i in train_img_name:
            images = nib.load(train_img_dir + i)
            
            images_array = images.get_fdata(dtype=np.float32) # shape : (z, y, x)
            # print(images.shape)
            
            split_name = i.split('.')
            new_name = split_name[0]

            if images_array[:,:,0].shape != (args.image_size, args.image_size):
                scale1 = args.image_size / images_array[:,0,0].shape
                scale2 = args.image_size / images_array[0,:,0].shape
                scale3 = 1.0
                scale_list = [scale1, scale2, scale3]

                images_array = zoom(images_array, scale_list, order=0)

            intercept = images.dataobj.inter
            slope = images.dataobj.inter

            images_array = images_array * slope + intercept

            if not args.window_width == None and args.window_level == None:
                image_min = args.window_level - (args.window_width / 2.0)
                image_max = args.window_level + (args.window_width / 2.0)

                # If image pixel is over max value or under min value, threshold to max and min
                images_array = np.clip(images_array, image_min, image_max)
            else:
                image_min = -1024.0
                image_max = 3071.0

                # If image pixel is over max value or under min value, threshold to max and min
                images_array = np.clip(images_array, image_min, image_max)

            if args.normalization == 'mean':
                mean, std = np.mean(images_array), np.std(images_array)
                images_array = (images_array - mean) / std
            elif args.normalization == 'max':
                min, max = np.min(images_array), np.max(images_array)
                images_array = (images_array - min) / (max - min)
            
            for index in range(images_array.shape[0]):
                
                if args.view == 'coronal':
                    image = images_array[index, :, :]  
                elif args.view == 'axial':
                    image = images_array[:, index, :]  
                elif args.view == 'sagittal':
                    image = images_array[:, :, index]  
                            
                filename = new_name + '_%02d.npy'%(index+1)
                np.save(args.save_dir + filename, image)
  