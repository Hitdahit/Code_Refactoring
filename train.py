import sys
import numpy as np
import pandas as pd

from tqdm import tqdm as tqdm

import torch
#device, dataloader

def train_one_epoch(args, epoch, scheduler=True):
    #model, dataloader, fn_loss, optimizer, device, epoch, print_freq, batch_size, use_amp
    args.model.train()

    if args.use_amp is True:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    log_header =  f"Epoch: [{epoch}]"

    for data in args.evaluation.metric_logger.log_every(args.train_dataloader, args.print_freq, log_header):
        x = data['image'].float().to(args.device)
        y = data['label'].long().to(args.device)
        
        y_pred = args.model(x)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            y_pred = args.model(x)
            loss_value = args.loss(y_pred, y)

        if args.use_amp is True:
                scaler.scale(loss_value).backward()
                scaler.step(args.optimizer)
                scaler.update()
                args.optimizer.zero_grad()

        else:
            args.optimizer.zero_grad()
            loss_value.backward()
            args.optimizer.step()
        
        loss_value = loss_value.item()

        if scheduler is True and args.scheduler is not None:
            args.scheduler.step()
        elif scheduler is True and args.scheduler is None:
            raise Exception('from train: you need to set your scheduler in your settings file')
        
        args.evaluation.metric_logger.update(loss=loss_value, lr=args.optimizer.param_groups[0]["lr"])
        args.evaluation.execute(y_pred, y)

        args.save_config.add_train_log('loss', epoch, scalar=loss_value)
        # for (i, j) in zip(args.evaluation.name, args.evaluation.metric_result):
        #     args.save_config.add_train_log(i, epoch, scalar=j)
    
    for i in args.evaluation.name: 
        df = pd.DataFrame.from_dict(args.evaluation.metric_result_dict)
        metric_val_mean = df.loc[df['name'] == i]['metric result'].mean()
        args.save_config.add_train_log(i, epoch, scalar=metric_val_mean)

    args.evaluation.name_list = []
    args.evaluation.metric_result = []
    args.evaluation.metric_result_dict = {'name': args.evaluation.name_list, 'metric result': args.evaluation.metric_result}
    
    return {k : round(meter.global_avg) for k, meter in args.evaluation.metric_logger.meters.items()}


@torch.no_grad()
def valid_one_epoch(args, epoch, scheduler=False):
    args.model.eval()
    
    log_header = 'TEST:'

    for data in args.evaluation.metric_logger.log_every(args.valid_dataloader, args.print_freq, log_header):
        x = data['image'].float().to(args.device)
        y = data['label'].long().to(args.device)

        y_pred = args.model(x)

        loss = args.loss(y_pred, y)
        loss_value = loss.item()

        if scheduler is True and args.scheduler is not None:
            scheduler.step()
        elif scheduler is True and args.scheduler is None:
            raise Exception('from validation: you need to set your scheduler in your settings file')
        args.evaluation.metric_logger.update(loss=loss_value)

        args.evaluation.metric_logger.update(loss=loss_value, lr=args.optimizer.param_groups[0]["lr"])
        args.evaluation.execute(y_pred, y)

        args.save_config.add_valid_log('loss', epoch, scalar=loss_value)

        for (i, j) in zip(args.evaluation.name, args.evaluation.metric_result):
            args.save_config.add_valid_log(i, epoch, scalar=j)
            
    args.evaluation.name_list = []
    args.evaluation.metric_result = []            
    args.evaluation.metric_result_dict = {'name': args.evaluation.name_list, 'metric result': args.evaluation.metric_result}

    return {k: round(meter.global_avg, 7) for k, meter in args.evaluation.metric_logger.meters.items()} 



