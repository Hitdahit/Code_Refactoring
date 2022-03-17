import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data_loader import *


def MC_metric(y, yhat):

    acc_targets = []
    acc_outputs = []

    y_temp = y
    for t in y_temp.view(-1,1).cpu():
        acc_targets.append(t.item()) 

    _, yhat_temp = torch.max(yhat, 1)
    for u in yhat_temp.view(-1,1).cpu():
        acc_outputs.append(u.item())

    cor = 0
    for i in range(len(acc_targets)):
        if acc_outputs[i] == acc_targets[i]:
            cor += 1

    acc = cor/len(acc_outputs)

    return acc


def multiclass_classification_train(args):
    '''
    multiclass classification training&validation 함수
    input: args
    output: train_loss, train_metric, val_loss, val_metric
    '''
    # data loading
    train_dataloader = get_loader(args, mode='train')
    val_dataloader = get_loader(args, mode='valid')
    
    # 텐서보드를 사용하기 위한 SummaryWriter 설정, log 파일을 저장할 경로
    writer_train = SummaryWriter(logs_dir=os.path.join(logs_dir, 'train'))
    writer_val = SummaryWriter(logs_dir=os.path.join(logs_dir, 'val'))

    # epoch 마다 하나씩 추가될 list들
    train_loss = []
    train_metric = []
    val_loss = []
    val_metric = []
    
    model = args.model
    loss_function = args.loss()
    optimizer = args.optimizer
    model.to(args.device) # model이 GPU에서 돌아가도록 설정

    for epoch in range(args.epochs): # 0 ~ (epochs-1) 번의 for 문을 실행. for문의 index = epoch
        model.train()
        loss_epoch = []
        metric_epoch = 0
        dataset_count = 0 # accuracy 계산을 위한 data 갯수 count

        for i, data in train_dataloader: # enumerate -> in
            x = data['image'].to(device)
            y = data['label'].to(device) 

            # forward
            yhat = model(x)
            loss = loss_function(yhat, y)
            metric = MC_metric(y, yhat)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_temp = y
            for t in y_temp.view(-1,1).cpu():
                train_acc_targets.append(t.item()) 

            _, yhat_temp = torch.max(yhat, 1)
            for u in yhat_temp.view(-1,1).cpu():
                train_acc_outputs.append(u.item())

            accuracy = accuracy_function(train_acc_outputs, train_acc_targets)
            loss_epoch.append(loss.detach().cpu().numpy())
            metric_epoch += metric
            dataset_count += x.shape[0]

        scheduler.step()

        # Tensorboard save
        writer_train.add_scalar('loss', np.mean(train_loss_epoch), epoch + 1)
        writer_train.add_scalar('accuracy', metric_epoch / dataset_count, epoch + 1)
        
        train_loss.append(np.mean(loss_epoch)) # 1 epoch 마다 train loss 추가
        train_metric.append(metic_epoch / dataset_count) # 1 epoch 마다 train metric 추가
        #한 epoch당 train loss, train acc 출력
        print(f'Epoch : {epoch + 1} Train Loss : {np.mean(train_loss_epoch)} Accuracy : {metric_epoch / train_dataset_count}')
        
        # validation
        with torch.no_grad():
            model.eval()
            val_loss_epoch = []
            val_accuracy_epoch = 0 # iter당 accuracy 계산값
            val_dataset_count = 0 # accuracy 계산을 위한 data 갯수 count

            for data in val_dataloader: # enumerate -> in
                # forward pass
                x = data['image'].to(device)
                y = data['label'].to(device) # multilabel_one_hot_encoding

                yhat = model(x)

                loss = loss_function(yhat, y)
                metric = MC_metric(y, yhat)

                loss_epoch.append(loss.detach().cpu().numpy())
                metric_epoch += metric
                dataset_count += x.shape[0]
        
        # Tensorboard save
        writer_val.add_scalar('loss', np.mean(loss_epoch), epoch + 1)
        writer_val.add_scalar('accuracy', metric_epoch / dataset_count, epoch + 1) #accuracy로 변경

        val_loss.append(np.mean(loss_epoch))
        val_metric.append(metric_epoch / dataset_count)
        #한 epoch당 val loss, val acc 출력
        print(f'Epoch : {epoch} Val Loss : {np.mean(val_loss_epoch)} Accuaracy : {val_accuracy_epoch / val_dataset_count} ')
        
        # model, optimizer save all epoch
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
       "./%s/model_epoch%d.pth.tar.gz" % (checkpoint_dir, epoch))
        
    writer_train.close()
    writer_val.close()

    return train_loss, train_accuarcy, val_loss, val_accuarcy
