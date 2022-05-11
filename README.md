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

​	i. Fork this repository to your private Repository
		![캡처](/imgs/fork.png)

​	ii. clone your repository to your workspace
		
		git clone {your repository url}

   iii. Setup your environment with configuration file

​		
	 iii-1. Anaconda

​	 iii-2. Docker

		---------------
		Dockerfile.ts
		---------------
		FROM tensorflow/tensorflow:latest-gpu
		RUN apt-get update && apt-get install -y apt-utils sudo vim libgl1-mesa-glx libgdcm-tools python3-pip 
		RUN apt-get update --fix-missing
		RUN pip3 install --upgrade pip
		RUN pip3 install jupyter_contrib_nbextensions torch torchsummary torchvision tensorboardX albumentations numpy opencv-python 			SimpleITK pydicom==2.1.1 Pillow==8.0.1 nibabel scipy matplotlib tqdm sklearn ipywidgets
		---------------

​	iv. version changing (for your ablation study)
	![캡처](/imgs/for_ablation.png)


### 2. Examples

​	i. How to start training

​		python train.py --t {txt files} --v{version number}

​	ii. 

​	iii. How to evaluate trained model 

### 3. Models
We provide 4 well-known CNN models (ResNet, VGG, DenseNet, EfficientNet) and each models have different number of layers. You can choose the architecture based on your dataset.

    i. ResNet : We have implemented 5 different architectures of ResNet such as ResNet18, ResNet34, ResNet50, Resnet101, ResNet152. 

    ii. VGG : We have implemented 4 different architectures of VGG such as VGG11, VGG13, VGG16, VGG19 with batch normalization.

    iii. DenseNet : We have implemented 3 different architectures of DenseNet such as DenseNet121, DenseNet169, DenseNet201.

    iv. EfficientNet : 

	v. SENet : SE Block..?

	vi. GoogLeNet : 

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
	소화기 내시경 동영상에서 추출한 '이미지' 기반으로 학습합니다.

	- Preprocessing: cropping
    	- 내시경 동영상에서 바로 추출한 영상은 640*480 혹은 1920*1080으로 이 경우엔 환자 정보들이 영상에 그대로 남은 상태입니다.
    	- 그러므로 cropping을 진행합니다.
  	- Image Normalization: 
    	- min_max_scaling 
    	- (TBD)
  	- 
	
​	v. ENT
    ENT는 수술장 비디오를 이용해 연구 하는 팀으로 3channel(RGB) Video Data를 사용합니다. 
​    Data Format은 Video를 Raw Frame(.png)으로 나눈 후 Clip 단위로 데이터를 구성하거나, MP4 Format을 사용합니다.
​    기본적인 전처리는 ImageNet 데이터 전처리와 동일합니다. 

    - Image Normalize : mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375] or image / 255.0
    
    - Image Resize : (1920 * 1080) -> (864 * 480) 

    - Frame Interval : 1 (변경 가능)

    - Clip Length : 8 frames(Video Segmenation), 32 frames(Video Recognition) 

    - Format Shape : N(Batch), C(Channels), T(Times), H(Height), W(Weidth)

** Note that supporting models in this repository are designed for 2 dimensional images.

​	if you want 3 dimensional images, try __

### 5. Supporting Extensions

​	i. torch.cuda.amp

​	ii. Slack alarm (by wandb)

### 6. How to Contribute 

​        

