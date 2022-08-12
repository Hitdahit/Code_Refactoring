FROM mi2rl/mi2rl_image:201130_tf1.15.0

RUN apt-get update
RUN apt-get install -y sudo vim
RUN python3 install --upgrade pip
RUN python3 install jupyter_contrib_nbextensions torch torchsummary torchvision tensorboardX albumentations numpy opencv-python 			
RUN python3 -m pip install SimpleITK pydicom==2.1.1 Pillow==8.0.1 nibabel scipy matplotlib tqdm sklearn ipywidgets

RUN addgroup --gid 1000001 gusers
RUN adduser --uid 1000UID --gid 1000001 --disabled-password --gecos '' #UserName#
RUN adduser #UserName# sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER #UserName#
