FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y apt-utils sudo vim libgl1-mesa-glx libgdcm-tools python3-pip 
RUN apt-get update --fix-missing
RUN python3 install --upgrade pip
RUN python3 install jupyter_contrib_nbextensions torch torchsummary torchvision tensorboardX albumentations numpy opencv-python 			
RUN python3 -m pip install SimpleITK pydicom==2.1.1 Pillow==8.0.1 nibabel scipy matplotlib tqdm sklearn ipywidgets
