from statistics import mode
import torch
import torch.nn as nn
from model.ResNet import *
from model.VGG import *
from model.DenseNet import *
#from efficientnet_pytorch import EfficientNet


# ResNet  18, 34, 50, 101, 152

class ResNet(nn.Module):
    def __init__(self, n_classes, model_size, pretrained = False):
        super(ResNet, self).__init__()
        
        if model_size == 18:
            self.pretrained_model = resnet18(pretrained = pretrained, num_classes = n_classes)
                
        elif model_size == 34:
            #self.pretrained_model = resnet34(pretrained = pretrained, num_classes = n_classes)
            self.pretrained_model = resnet34()
                
        elif model_size == 50:
            self.pretrained_model = resnet50(pretrained = pretrained, num_classes = n_classes)
                
        elif model_size == 101:
            self.pretrained_model = resnet101(pretrained = pretrained, num_classes = n_classes)
                
        elif model_size == 152:
            self.pretrained_model = resnet152(pretrained = pretrained, num_classes = n_classes)
                
    def forward(self, imgs):
        output = self.pretrained_model(imgs)
        return output
        
        
# VGG  11 13 16 19

class VGG_Classifier(nn.Module):
    def __init__(self, n_classes, model_size, pretrained = False):
        super(VGG_Classifier, self).__init__()
        
        if model_size == 11:
            self.pretrained_model = VGG.vgg11_bn(pretrained = pretrained, num_classes = n_classes)
                
        elif model_size == 13:
            self.pretrained_model = VGG.vgg13_bn(pretrained = pretrained, num_classes = n_classes)
                
        elif model_size == 16:
            self.pretrained_model = VGG.vgg16_bn(pretrained = pretrained, num_classes = n_classes)
                
        elif model_size == 19:
            self.pretrained_model = VGG.vgg19_bn(pretrained = pretrained, num_classes = n_classes)
                
    def forward(self, imgs):
        output = self.pretrained_model(imgs)
        return output
    
    
# DenseNet  121 169 201

class DenseNet_Classifier(nn.Module):
    def __init__(self, n_classes, model_size, pretrained = False):
        super(DenseNet_Classifier, self).__init__()
    
        if model_size == 121:
            self.pretrained_model = DenseNet.DenseNet_121(pretrained = pretrained, num_classes = n_classes)
                
        elif model_size == 169:
            self.pretrained_model = DenseNet.DenseNet_169(pretrained = pretrained, num_classes = n_classes)
                
        elif model_size == 201:
            self.pretrained_model = DenseNet.DenseNet_201(pretrained = pretrained, num_classes = n_classes)
                
    def forward(self, imgs):
        output = self.pretrained_model(imgs)
        return output
    
    
'''  
# EfficientNet

class EfficientNet_model(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_model, self).__init__()
        self.model = EfficientNet.from_name('efficientnet-b3', num_classes = n_classes)
        # self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.model._avg_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, imgs):
        output = self.model(imgs)

        return output

'''