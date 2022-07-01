import sys
import numpy as np
from tqdm import tqdm as tqdm

import torch

def train_epoch_classification(model, dataloader, fn_loss, optimizer, device, epoch, print_freq, batch_size, use_amp):
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metric_logger = MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header =  f"Epoch: [{epoch}]"

    for data in metric_logger.log_every(dataloader, print_freq, header):
        x = data['image'].float().to(device)
        y = data['label'].long().to(device)

        y_pred = model(x)

        with torch.cuda.amp.autocast(enabled=use_amp):
            y_pred = model(x)
            loss = fn_loss(y_pred, y)

        
        loss_value = loss.item()

        if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return {k : round(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def valid_epoch_classification(model, dataloader, fn_loss,  device, metric, print_freq, batch_size):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    for data in metric_logger.log_every(dataloader, print_freq, header):
        x = data['image'].float().to(device)
        y = data['label'].long().to(device)

        y_pred = model(x)

        loss = fn_loss(y_pred, y)
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        accuracy = metric(y_pred, y)
        metric_logger.update(acc=accuracy)

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}