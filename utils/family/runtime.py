import os
import re
import sys
import time
import datetime
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.utils.SummaryWriter as SummaryWriter


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


class Metrics(nn.Module):
    def __init__(self, name=None, threshold=0.5, eps=1e-7, beta=1, activation=None):
        super().__init__()
        self._name = name
        self.threshold = threshold
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        else:
            return self._name



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

class Fscore(Metrics):
    def __init__(self, **kwargs):
        super(Fscore, self).__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return f_score(y_pr, y_gt, eps=self.eps, beta=self.beta, threshold=self.threshold)

class Accuracy(Metrics):
    def __init__(self, **kwargs):
        super(Accuracy, self).__init__(**kwargs)
        
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return accuracy(y_pr, y_gt, threshold=self.threshold)


class Recall(Metrics):
    def __init__(self, **kwargs):
        super(Recall, self).__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return recall(y_pr, y_gt, eps=self.eps, threshold=self.threshold)


class Precision(Metrics):
    def __init__(self, **kwargs):
        super(Precision, self).__init__(**kwargs)
        
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return precision(y_pr, y_gt, eps=self.eps, threshold=self.threshold)
    
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
Checkpoint & Log saver
'''
class Saver():
    def __init__(self, log_library='wandb', log_dir, ckpt_dir, experiment_name):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.ckpt_dir = ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.experiment_name = experiment_name    
        if 'wandb' in log_library:
            self.logger_train = # wandb setup
            self.logger_valid = # wandb setup
        
        elif 'tensorboard' in log_library:
            self.logger_train = SummaryWriter(logs_dir=os.path.join(self.log_dir, self.experiment_name, 'train'))
            self.logger_valid = SummaryWriter(logs_dir=os.path.join(self.log_dir, self.experiment_name, 'valid'))
        
    def save_checkpoint(self, net, optimizer, epoch):
        '''
        부가 기능 추가 가능 (ex acc등 적기)
        key값 반드시 lower case로 적을 것.
        '''
        torch.save({'model': net.state_dict(), 'optimizer':optimizer.state_dict()}, '%s/%s/%d.pth'%(self.ckpt_dir, self.experiment_name, epoch))

