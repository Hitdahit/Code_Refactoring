import sys
import torch
from tqdm import tqdm as tqdm
import numpy as np

def format_logs(logs):
    str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
    s = ", ".join(str_logs)
    return s

def train_epoch_classification(
    model,
    dataloader,
    fn_loss,
    metric,
    optimizer,
    device
):
    model.train()

    logs = {}
    loss_value = []
    acc_scores = []
    
    with tqdm(
        dataloader,
        desc='train',
        file=sys.stdout,
        disable= not True,
        ncols=100
    ) as iterator:
        for data in iterator:
            x = data['image'].float().to(device)
            y = data['label'].long().to(device)

            y_pred = model(x)

            optimizer.zero_grad()

            loss = fn_loss(y_pred, y).to(device)
            loss_value.append(loss.item())
            loss_logs = {'loss' : np.mean(loss_value)}
            logs.update(loss_logs)
            loss.backward()

            acc = metric(y_pred, y).to(device)
            acc_scores.append(acc.item())
            metric_logs = {'acc' : np.mean(acc_scores)}
            logs.update(metric_logs)

            optimizer.step()

            s = format_logs(logs)
            iterator.set_postfix_str(s)
    
    return logs


@torch.no_grad()
def valid_epoch_classification(
    model,
    dataloader,
    fn_loss,
    metric,
    device
):
    model.eval()

    logs = {}
    loss_value = []
    acc_scores = []

    with tqdm(
        dataloader,
        desc='valid',
        file=sys.stdout,
        disable= not True,
        ncols=100
    ) as iterator:
        for data in iterator:
            x = data['image'].float().to(device)
            y = data['label'].long().to(device)

            y_pred = model(x)

            loss = fn_loss(y_pred, y).to(device)
            loss_value.append(loss.item())
            loss_logs = {'loss' : np.mean(loss_value)}
            logs.update(loss_logs)

            acc = metric(y_pred, y).to(device)
            acc_scores.append(acc.item())

            metric_logs = {'acc' : np.mean(acc_scores)}
            logs.update(metric_logs)

            s = format_logs(logs)
            iterator.set_postfix_str(s)
    
    return logs