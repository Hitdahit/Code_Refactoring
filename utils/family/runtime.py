import os
import re
import sys
import time
import datetime
import wandb
import math
from torch.optim.lr_scheduler import _LRScheduler
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

'''
Metrics
'''
    
class Metrics():
    def __init__(self, batch_size, activation=None, threshold=0.5, name=None, eps=1e-7, beta=1):
        super().__init__()

        self.name = name

        self.activation = Activation(activation)

        self.threshold = threshold
        
        self.eps = eps
        self.beta = beta
        
        self.metric_logger = MetricLogger(delimiter="  ", n=batch_size)
        
        for i in self.name:
            self.metric_logger.add_meter(i, SmoothedValue(window_size=1, fmt='{value:.6f}')) 

        self.metric_result=[]
    def execute(self, pred, gt):
        pred = self.activation(pred)
        
        for i in self.name: 
            res = getattr(self, i)(pred, gt)
            dic = {i:res}

            self.metric_result.append(res)
            self.metric_logger.update(**dic)
        
        
    
    @property
    def __name__(self):
        if self.name is None:
            name = self.__class__.__name__
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        else:
            return self.name

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
    아직 지원 x
    '''
    def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
        '''
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        ---------
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html

        title:         Title for the heatmap. Default is None.
        '''


        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]

        if group_names and len(group_names)==cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            #Accuracy is sum of diagonal divided by total observations
            accuracy  = np.trace(cf) / float(np.sum(cf))

            #if it is a binary confusion matrix, show some more stats
            if len(cf)==2:
                #Metrics for Binary Confusion Matrices
                precision = cf[1,1] / sum(cf[:,1])
                recall    = cf[1,1] / sum(cf[1,:])
                f1_score  = 2*precision*recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy,precision,recall,f1_score)
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""


        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize==None:
            #Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks==False:
            #Do not show categories if xyticks is False
            categories=False


        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + stats_text)

        else:
            plt.xlabel(stats_text)

        if title:
            plt.title(title)
            
######################## ACTIVATIONS #############################
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
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.argmax(x, dim=self.dim)
    

######################## SAVE LOGS & CHECKPOINTS #############################
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
            self.log_library = 'wandb'
            self.logger_train = None# wandb setup
            self.logger_valid = None# wandb setup
        
        elif 'tensorboard' in log_library:
            self.log_library = 'tensorboard'
            self.logger_train = SummaryWriter(log_dir=os.path.join(self.log_dir, 'train'))
            self.logger_valid = SummaryWriter(log_dir=os.path.join(self.log_dir, 'valid'))
        
    
    def add_train_log(self, tag, step, scalar=None, image=None):
        '''
        tag -> str
        scalar -> numpy
        step -> int
                log epoch by epoch-> epoch
                log step by step -> epoch*(Whole_Dataset / Batch_Size)
        image -> numpy 
        '''
        if 'wandb' in self.log_library:
            pass
        elif 'tensorboard' in self.log_library:
            if scalar is not None:
                self.logger_train.add_scalar(tag, scalar, step)
            if image is not None:
                self.logger_train.add_image(tag, image, step)


    def add_valid_log(self, mode, tag, step, scalar=None, image=None):
        if 'wandb' in self.log_library:
            pass
        elif 'tensorboard' in self.log_library:
            if scalar is not None:
                self.logger_valid.add_scalar(tag, scalar, step)
            if image is not None:
                self.logger_valid.add_image(tag, image, step)

    def save_checkpoint(self, net, optimizer, epoch):
        '''
        부가 기능 추가 가능 (ex acc 기록 등)
        key값 반드시 lower case로 적을 것.
        '''
        torch.save({'net': net.state_dict(), 'optim':optimizer.state_dict()}, '%s/%s/%d.pth'%(self.ckpt_dir, self.experiment_name, epoch))

######################## LOSS #############################
'''
if you want to custom your loss,
then implement your loss in 'cutom_loss' class like code below.

ex.
class FocalLoss(nn.Module):
    def__init__(self, weight=None, gamma=2.f, reduction='none'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, gt):
        log_prob = F.log_softmax(pred, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(((1-prob)**self.gamma)*log_prob, gt, weight=self.weight, reduction=self.reduction)
'''


class custom_loss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward():
        pass

######################## LR_SCHEDULER #############################
'''
if you want to custom your lr scheduler,
then implement your loss in 'cutom_LRScheduler' class like code below.

ex. https://github.com/gaussian37/pytorch_deep_learning_models/blob/master/cosine_annealing_with_warmup/cosine_annealing_with_warmup.py

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
'''
class custom_LRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch):
        super(custom_LRScheduler, self).__init__(optimizer, last_epoch)
        pass

    def _get_lr(self):
        '''
        implement your equation to change your learning rate
        '''
        pass

    def step(self):
        '''
        use this funtion in your training loop
        '''
        pass

   
################################# Do Not Touch Here #######################################

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        # n is batch_size
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t", n=1):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter  = delimiter
        self.n = n

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(value=v, n=self.n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)


    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))