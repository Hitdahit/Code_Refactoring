import sys
import numpy as np
from tqdm import tqdm as tqdm

import torch

def format_logs(logs):
    str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
    s = ", ".join(str_logs)
    return s


def train_epoch_classification(model, dataloader, fn_loss, metrics, optimizer, use_amp, device):
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logs = {}
    loss_meter = []
    metrics_meters = {metric.__name__:[] for metric in metrics}
    
    with tqdm(dataloader, desc='train', file=sys.stdout, disable= not True, ncols=100) as iterator:
        for data in iterator:
            x = data['image'].float().to(device)
            y = data['label'].long().to(device)

            y_pred = model(x)

            with torch.cuda.amp.autocast(enabled=use_amp):
                y_pred = model(x)
                loss = fn_loss(y_pred, y)

            loss_meter.append(loss.item())
            loss_log = {'loss' : np.mean(loss_meter)}
            logs.update(loss_log)
            
            for metric_fn in metrics:
                metric_value = metric_fn(y_pred, y)
                metrics_meters[metric_fn.__name__].append(metric_value.item())
            metrics_logs = {k:np.mean(v) for k, v in metrics_meters.items()}
            logs.update(metrics_logs)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            s = format_logs(logs)
            iterator.set_postfix_str(s)
    
    return logs


@torch.no_grad()
def valid_epoch_classification(model, dataloader, fn_loss, metrics, device):
    model.eval()

    logs = {}
    loss_meter = []
    metrics_meters = {metric.__name__:[] for metric in metrics}

    with tqdm(dataloader, desc='valid', file=sys.stdout, disable= not True, ncols=100) as iterator:
        for data in iterator:
            x = data['image'].float().to(device)
            y = data['label'].long().to(device)

            y_pred = model(x)

            loss = fn_loss(y_pred, y)
            loss_meter.append(loss.item())
            loss_log = {'loss' : np.mean(loss_meter)}
            logs.update(loss_log)

            for metric_fn in metrics:
                metric_value = metric_fn(y_pred, y)
                metrics_meters[metric_fn.__name__].append(metric_value.item())
            metrics_logs = {k:np.mean(v) for k, v in metrics_meters.items()}
            logs.update(metrics_logs)

            s = format_logs(logs)
            iterator.set_postfix_str(s)
    
    return logs


'''
ex)
metrics = [utils.metrics.Accuracy(threshold=0.5, activation='sigmoid'), 
           utils.metrics.Recall(), utils.metrics.Precision]

train_logs = []
valid_logs = []

for epoch in range(start_epoch, args.EPOCHS):
    print('\nEpoch: {}, LR: {}'.format(epoch, optimizer.param_groups[0]['lr']))
    
    train_log = train_epoch_classification(
        model,
        trainloader,
        fn_loss,
        metrics,
        optimizer,
        ues_amp,
        device
    )

    valid_log = valid_epoch_classification(
        model,
        validloader,
        fn_loss,
        metrics,
        device
    )

    scheduler.step()

    train_logs.append(train_log)
    valid_logs.append(valid_log)
'''