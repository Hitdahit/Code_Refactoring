from statistics import mode
import torch
import torch.nn as nn
from model import ResNet, VGG, DenseNet
#from efficientnet_pytorch import EfficientNet


# ResNet  18, 34, 50, 101, 152

class ResNet_Classifier(nn.Module):
    def __init__(self, n_classes, model_size, pretrained = False):
        super(ResNet, self).__init__()
        
        if model_size == 18:
            self.pretrained_model = ResNet.resnet18(pretrained = pretrained, num_classes = n_classes)
            if pretrained is not True:
                self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
                
        elif model_size == 34:
            self.pretrained_model = ResNet.resnet34(pretrained = pretrained, num_classes = n_classes)
            if pretrained is not True:
                self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
                
        elif model_size == 50:
            self.pretrained_model = ResNet.resnet50(pretrained = pretrained, num_classes = n_classes)
            if pretrained is not True:
                self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
                
        elif model_size == 101:
            self.pretrained_model = ResNet.resnet101(pretrained = pretrained, num_classes = n_classes)
            if pretrained is not True:
                self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
                
        elif model_size == 152:
            self.pretrained_model = ResNet.resnet152(pretrained = pretrained, num_classes = n_classes)
            if pretrained is not True:
                self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
                
    def forward(self, imgs):
        output = self.pretrained_model(imgs)
        return output
        
        
# VGG  11 13 16 19

class VGG_Classifier(nn.Module):
    def __init__(self, n_classes, model_size, pretrained = False):
        super(VGG_Classifier, self).__init__()
        
        if model_size == 11:
            self.pretrained_model = VGG.vgg11_bn(pretrained = pretrained, num_classes = n_classes)
        
            if pretrained is not True:
                self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
                
        elif model_size == 13:
            self.pretrained_model = VGG.vgg13_bn(pretrained = pretrained, num_classes = n_classes)
        
            if pretrained is not True:
                self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
                
        elif model_size == 16:
            self.pretrained_model = VGG.vgg16_bn(pretrained = pretrained, num_classes = n_classes)
        
            if pretrained is not True:
                self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
                
        elif model_size == 19:
            self.pretrained_model = VGG.vgg19_bn(pretrained = pretrained, num_classes = n_classes)
        
            if pretrained is not True:
                self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
                
    def forward(self, imgs):
        output = self.pretrained_model(imgs)
        return output
    
    
# DenseNet  121 169 201

class DenseNet_Classifier(nn.Module):
    def __init__(self, n_classes, model_size, pretrained = False):
        super(DenseNet_Classifier, self).__init__()
    
        if model_size == 121:
            self.pretrained_model = DenseNet.DenseNet_121(pretrained = pretrained, num_classes = n_classes)
        
            if pretrained is not True:
                self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
                
        elif model_size == 169:
            self.pretrained_model = DenseNet.DenseNet_169(pretrained = pretrained, num_classes = n_classes)
        
            if pretrained is not True:
                self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
                
        elif model_size == 201:
            self.pretrained_model = DenseNet.DenseNet_201(pretrained = pretrained, num_classes = n_classes)
        
            if pretrained is not True:
                self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
                
    def forward(self, imgs):
        output = self.pretrained_model(imgs)
        return output
    
    
'''
    

class ResNet34_Classifier(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(ResNet34_Classifier, self).__init__()
        self.pretrained_model = ResNet.resnet34(pretrained = pretrained, num_classes = n_classes)
        
        if pretrained is not True:
            self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
            
    def forward(self, imgs):
        res_out = self.pretrained_model(imgs)
        return res_out
    
class ResNet50_Classifier(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(ResNet50_Classifier, self).__init__()
        self.pretrained_model = ResNet.resnet50(pretrained = pretrained, num_classes = n_classes)
        
        if pretrained is not True:
            self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
            
    def forward(self, imgs):
        res_out = self.pretrained_model(imgs)
        return res_out
    
class ResNet101_Classifier(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(ResNet101_Classifier, self).__init__()
        self.pretrained_model = ResNet.resnet101(pretrained = pretrained, num_classes = n_classes)
        
        if pretrained is not True:
            self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
            
    def forward(self, imgs):
        res_out = self.pretrained_model(imgs)
        return res_out
    
    
# VGG

class vgg16_model(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(vgg16_model, self).__init__()
        self.pretrained_model = VGG.vgg16_bn(pretrained = pretrained, num_classes = n_classes)
        
        if pretrained is not True:
            self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
           
    def forward(self, imgs):
        output = self.pretrained_model(imgs)
        return output
    
class vgg19_model(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(vgg19_model, self).__init__()
        self.pretrained_model = VGG.vgg19_bn(pretrained = pretrained, num_classes = n_classes)
        
        if pretrained is not True:
            self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, imgs):
        output = self.pretrained_model(imgs)
        return output
    
    
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
    
# DenseNet

class DenseNet121_Classifier(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(DenseNet121_Classifier, self).__init__()
        self.pretrained_model = DenseNet.DenseNet_121(pretrained = pretrained, num_classes = n_classes)
        
        if pretrained is not True:
            self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
            
    def forward(self, imgs):
        output = self.pretrained_model(imgs)
        return output
    
class DenseNet169_Classifier(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(DenseNet169_Classifier, self).__init__()
        self.pretrained_model = DenseNet.DenseNet_169(pretrained = pretrained, num_classes = n_classes)
        
        if pretrained is not True:
            self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
            
    def forward(self, imgs):
        output = self.pretrained_model(imgs)
        return output

'''