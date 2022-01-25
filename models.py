import torch
import torch.nn as nn
import ResNet, VGG, DenseNet
from efficientnet_pytorch import EfficientNet

'''
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)

efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')


model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=True)
'''

# ResNet

class ResNet34_Classifier(nn.Module):
    def __init__(self, n_classes, pretrained =False):
        super(ResNet50_Classifier, self).__init__()
        self.pretrained_model = ResNet.resnet34(pretrained=pretrained, num_classes = n_classes)
        
        if pretrained is not True:
            self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
            
    def forward(self, imgs):
        res_out = self.pretrained_model(imgs)
        return res_out
    
class ResNet50_Classifier(nn.Module):
    def __init__(self, n_classes, pretrained =False):
        super(ResNet50_Classifier, self).__init__()
        self.pretrained_model = ResNet.resnet50(pretrained=pretrained, num_classes = n_classes)
        
        if pretrained is not True:
            self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
            
    def forward(self, imgs):
        res_out = self.pretrained_model(imgs)
        return res_out
    
class ResNet101_Classifier(nn.Module):
    def __init__(self, n_classes, pretrained =False):
        super(ResNet101_Classifier, self).__init__()
        self.pretrained_model = ResNet.resnet101(pretrained=pretrained, num_classes = n_classes)
        
        if pretrained is not True:
            self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
            
    def forward(self, imgs):
        res_out = self.pretrained_model(imgs)
        return res_out
    
    
# VGG

class vgg16_model(nn.Module):
    def __init__(self, n_classes):
        super(vgg16_model, self).__init__()
        self.model = VGG.vgg16_bn(pretrained=False,num_classes = n_classes)
        # self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, imgs):
        output = self.model(imgs)
        return output
    
class vgg19_model(nn.Module):
    def __init__(self, n_classes):
        super(vgg19_model, self).__init__()
        self.model = VGG.vgg19_bn(pretrained=False,num_classes = n_classes)
        # self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, imgs):
        output = self.model(imgs)
        return output
    
    
# EfficientNet

class EfficientNet_model(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_model, self).__init__()
        self.model = EfficientNet.from_name('efficientnet-b3',num_classes = n_classes)
        # self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.model._avg_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, imgs):
        output = self.model(imgs)

        return output
    
# DenseNet

class DenseNet121_Classifier(nn.Module):
    def __init__(self, n_classes, pretrained =False):
        super(DenseNet121_Classifier, self).__init__()
        self.pretrained_model = DenseNet.DenseNet_121(pretrained=pretrained, num_classes = n_classes)
        
        if pretrained is not True:
            self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
            
    def forward(self, imgs):
        output = self.pretrained_model(imgs)
        return output
    
class DenseNet169_Classifier(nn.Module):
    def __init__(self, n_classes, pretrained =False):
        super(DenseNet169_Classifier, self).__init__()
        self.pretrained_model = DenseNet.DenseNet_169(pretrained=pretrained, num_classes = n_classes)
        
        if pretrained is not True:
            self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
            
    def forward(self, imgs):
        output = self.pretrained_model(imgs)
        return output