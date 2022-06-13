import re
import torch
import torch.nn as nn

'''
Metrics
'''

class Activation(nn.Module):
    def __init__(self, name, **param):
        super.__init__()
        
        if name is None or name == 'identity':
            self.activation = nn.Identity(**param)
        elif 'custom' in name:
            self.activation = custom_activation(**param)
        elif callable(name): # __call__ 함수의 유무
            self.activation = name(**param)
        
        else:
            self.activation = getattr(sys.modules['torch.nn'], name)(**param)
        
    def forward(self, x):
        return self.activation(x)
    
class Metrics():
    def __init__(self, activation=None, threshold=0.5, name=None, eps=1e-7, **kwargs):
        self.activation = Activation(activation)
        self.eps = eps
        self.threshold = threshold
        self.metric = # 얘네 self.threshold, name,  eps, **kwargs 불러서 쓰게끔
    
    def forward(self, pred, gt):
        pred = self.activation(pred)
        return self.metric(pred, gt)
        
    
    
'''
Checkpoint & Log saver
'''
class Saver():
    

    
    
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
    
'''
Metrics:
    Select your wanted metircs from here or
    write your own metrics to evaluate your model
    
'''
def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

def accuracy(pr, gt, threshold=0.5):
    pr = _threshold(pr, threshold=threshold)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.size(0)
    return score


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None):
    pr = _threshold(pr, threshold=threshold)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score

def precision(pr, gt, eps=1e-7, threshold=None):
    pr = _threshold(pr, threshold=threshold)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score

def recall(pr, gt, eps=1e-7, threshold=None):
    pr = _threshold(pr, threshold=threshold)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return  score 