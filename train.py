import sys
import numpy as np
from tqdm import tqdm as tqdm

import torch
#device, dataloader

def train_one_epoch(args, epoch):
    #model, dataloader, fn_loss, optimizer, device, epoch, print_freq, batch_size, use_amp
    args.model.train()

    if args.use_amp is True:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    log_header =  f"Epoch: [{epoch}]"

    for data in args.evaluation.metric_logger.log_every(args.train_dataloader, args.print_freq, log_header):
        x = data['image'].float().to(args.device)
        y = data['label'].long().to(args.device)

        y_pred = args.model(x).to(args.device)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            y_pred = args.model(x)
            loss = args.loss(y_pred, y)


        loss_value = loss.item()

        if args.use_amp is True:
                scaler.scale(loss).backward()
                scaler.step(args.optimizer)
                scaler.update()
                args.optimizer.zero_grad()

        else:
            args.optimizer.zero_grad()
            loss.backward()
            args.optimizer.step()

        args.evaluation.metric_logger.update(loss=loss_value, lr=args.optimizer.param_groups[0]["lr"])
        args.evaluation.execute(y_pred, y)
    

    return {k : round(meter.global_avg) for k, meter in args.evaluation.metric_logger.meters.items()}


@torch.no_grad()
def valid_one_epoch(args, epoch):
    args.model.eval()
    
    log_header = 'TEST:'

    for data in args.evaluation.metric_logger.log_every(args.valid_dataloader, args.print_freq, log_header):
        x = data['image'].float().to(args.device)
        y = data['label'].long().to(args.device)

        y_pred = args.model(x)

        loss = args.loss(y_pred, y)
        loss_value = loss.item()
        args.evaluation.metric_logger.update(loss=loss_value)

        args.evaluation.metric_logger.update(loss=loss_value, lr=args.optimizer.param_groups[0]["lr"])
        args.evaluation.execute(y_pred, y)

    return {k: round(meter.global_avg, 7) for k, meter in args.evaluation.metric_logger.meters.items()} 



