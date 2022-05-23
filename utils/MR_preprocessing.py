import os
import cv2
import numpy as np
import SimpleITK as sitk

        
if __name__ == '__main__':      
    
    train_img_dir = 'directory of train image'
    train_msk_dir = 'directory of train mask'

    train_img_name = os.listdir('directory of train image')
    train_msk_name = os.listdir('directory of train mask')
    
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
            image = images_array[index, :, :]  # coronal view
            #image = images_array[:, index, :]  # axial view
            #image = images_array[:, :, index]  # sagittal view
                        
            filename = new_name + '_%02d.npy'%(index+1)
            np.save(train_img_dir + filename, image)
            
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
            mask = masks_array[index, :, :]  # coronal view
            #mask = masks_array[:, index, :]  # axial view
            #mask = masks_array[:, :, index]  # sagittal view
                        
            filename = new_name + '_%02d.npy'%(index+1)
            np.save(train_msk_dir + filename, mask)
    '''
  