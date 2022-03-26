# from MI2RL import Untitled

subtitle: MI2RL.exec()

---

The main features of this snippets are:

- Simple skeleton codes for medical image classification written by Pytorch
- Simple hyper parameter setting by text / yaml files
- Customized data loader for various medical imaging modalities
- Various preprocessing / Evaluation tools for medical images
- Easy to merge your own code 

## Table of Content

	1. Installation
 	2. Examples
 	3. Models
 	4. Supporting Modalities
 	5. Supporting Extensions
 	6. How to Contribute

### 1. Installation

​	i. How to Download

​	ii. How to setup your environment

​		ii-1. Anaconda

​		ii-2. Docker

​	iii. Sanity Check

### 2. Examples

​	i. How to start training

​		/python train.py --t {txt files} --v{version number}

​	ii. 

​	iii. How to evaluate trained model 

### 3. Models
We provide 4 well-known CNN models (ResNet, VGG, DenseNet, EfficientNet) and each models have different number of layers. You can choose the architecture based on your dataset.

    i. ResNet : (ResNet uses residual connections between the layers to solve gradient vanishing problem in deep neural networks. - 이런 소개 글 필요?) We have implemented 5 different architectures of ResNet such as ResNet18, ResNet34, ResNet50, Resnet101, ResNet152. 

    ii. VGG

    iii. DenseNet

    iv. EfficientNet

### 4. Supporting Modalities

​	i. Chest X-rays

		Chest X-rays Prepreocessing is divided with two methods.

		1. min-max scaling : First, load dicom image. Then change it to pixel array and change data type to float32. Second, If dicom image has 3 channel, change it to 1 channel and array to (image_size,image_size) by using squeeze(). Then resize image size you want. Third, execute min-max scaling with bits of dicom image.
		It is normalized from 0 to 1. If pixel value is over 1 or under 0, it must be changed to 1 or 0. Finally, check PhotometricInterpretation of dicom image. If it is MONOCHROME1, change it to MONOCHROME2.

		2. percentile : The previous two preprocessing processes of percentile are the same as those of min-max scaling. And execute percentile to reduce intensity of L,R mark. It is normalized from 0 to 1. If pixel value is over 1 or under 0, it must be changed to 1 or 0. Finally, check PhotometricInterpretation of dicom image. If it is MONOCHROME1, change it to MONOCHROME2.

		Note that X-ray dicom image has 8,12,16 bits.

​	ii. CT slices

		The previous two preprocessing precesses of CT slices are the same as those of Chest X-rays. And CT image has Rescale Intercept and Rescale Slope. Rescale Slope is multiplied and then Rescale Intercept is added to CT image array. It is mandantory preprocess task. Third, execute min-max scaling with bits of dicom image. It is normalized from 0 to 1. If pixel value is over 1 or under 0, it must be changed to 1 or 0. Finally, check PhotometricInterpretation of dicom image. If it is MONOCHROME1, change it to MONOCHROME2.

		Note that CT slices dicom image has 8,12,16 bits.

​	iii. MR slices

​	iv. Gastro Endoscopy

​	v. ENT 

** Note that supporting models in this repository are designed for 2 dimensional images.

​	if you want 3 dimensional images, try __

### 5. Supporting Extensions

​	i. torch.cuda.amp

​	ii. Slack alarm (by wandb)

### 6. How to Contribute 









