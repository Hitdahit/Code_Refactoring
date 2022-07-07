import os
import re
import sys
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
'''
Metrics
'''

class Activation(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name is None or name == 'identity':
            self.activation = nn.Identity()
        elif 'custom' in name:
            self.activation = custom_activation()
        elif callable(name): # __call__ 함수의 유무
            self.activation = name()
        
        else:
            self.activation = getattr(sys.modules['torch.nn'], name)(**param)
        
    def forward(self, x):
        return self.activation(x)
    
class Metrics():
    def __init__(self, activation=None, threshold=0.5, name=None, eps=1e-7, beta=1):
        self.activation = Activation(activation)
        
        self.threshold = threshold
        self.metric = getattr(self, name) #얘네 self.threshold, name,  eps, **kwargs 불러서 쓰게끔
        self.eps = eps
        self.beta = beta
        
    def execute(self, pred, gt):
        pred = self.activation(pred)
        return self.metric(pred, gt)
    
    '''
    Metrics:
        Select your wanted metircs from here or
        write your own metrics to evaluate your model
        
    '''
    def _threshold(self, x):
        if self.threshold is not None:
            return (x > self.threshold).type(x.dtype)
        else:
            return x
    
    def accuracy(self, pr, gt):
        pr = self._threshold(pr)
    
        tp = torch.sum(gt == pr, dtype=pr.dtype)
        score = tp / gt.size(0)
        return score
    
    
    def f_score(self, pr, gt):
        pr = self._threshold(pr)
    
        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp
    
        score = ((1 + self.beta ** 2) * tp + self.eps) / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.eps)
    
        return score
    
    def precision(self, pr, gt):
        pr = self._threshold(pr)
    
        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
    
        score = (tp + self.eps) / (tp + fp + self.eps)
    
        return score
    
    def recall(self, pr, gt):
        pr = self._threshold(pr)
    
        tp = torch.sum(gt * pr)
        fn = torch.sum(gt) - tp
    
        score = (tp + self.eps) / (tp + fn + self.eps)
    
        return  score 
        
    
    
'''
Checkpoint & Log saver
'''
class Saver():
    def __init__(self, log_dir, ckpt_dir, experiment_name, log_library='wandb'):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.ckpt_dir = ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            
        self.experiment_name = experiment_name    
        
        if 'wandb' in log_library:
            self.logger_train = None# wandb setup
            self.logger_valid = None# wandb setup
        
        elif 'tensorboard' in log_library:
            self.logger_train = SummaryWriter(log_dir=os.path.join(self.log_dir, self.experiment_name, 'train'))
            self.logger_valid = SummaryWriter(log_dir=os.path.join(self.log_dir, self.experiment_name, 'valid'))
        
    
    def add_log(self, ):
        pass
    def save_checkpoint(self, net, optimizer, epoch):
        '''
        부가 기능 추가 가능 (ex acc등 적기)
        key값 반드시 lower case로 적을 것.
        '''
        torch.save({'net': net.state_dict(), 'optim':optimizer.state_dict()}, '%s/%s/%d.pth'%(self.ckpt_dir, self.experiment_name, epoch))

    
    
#################################Detail Funtions#######################################
'''
Custom Activation:
    write your own custom activation function for logits
    
    ex 1)
        class ArgMax(nn.Module):
            def __init__(self, dim=None):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.argmax(x, dim=self.dim)
    
    ex 2)
        class Clamp(nn.Module):
            def __init__(self, min=0, max=1):
                super().__init__()
                self.min, self.max = min, max

            def forward(self, x):
                return torch.clamp(x, self.min, self.max)
'''
class custom_activation(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass
    
