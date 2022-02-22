import torch 
import torch.nn as nn # loss function, nn.Module
import torch.optim as optim # optimizer 사용
from torch.utils.tensorboard import SummaryWriter # 텐서보드 사용

import os # os.path.join 사용
import numpy as np

from sklearn.metrics import accuracy_score

# val = validation
def BC_metric(y, yhat):
    metric = accuracy_score(y, yhat)
    return metric

def binary_classification_train(args):
    '''
    binary classification training&validation 함수
    input: args
    output: train_loss, train_metric, val_loss, val_metric
    '''
    # data loading
    train_dataloader = get_loader(args)
    val_dataloader = get_loader(args)
    
    # 텐서보드를 사용하기 위한 SummaryWriter 설정, log 파일을 저장할 경로
    writer_train = SummaryWriter(logs_dir=os.path.join(args.log_dir, 'train'))
    writer_val = SummaryWriter(logs_dir=os.path.join(args.log_dir, 'val'))

    # epoch 마다 하나씩 추가될 list들
    train_loss = []
    train_metric = []
    val_loss = []
    val_metric = []
    
    model = args.model
    loss_function = args.loss
    optimizer = args.optimizer
    model.to(args.device) # model이 GPU에서 돌아가도록 설정

    for epoch in range(args.epochs): # 0 ~ (epochs-1) 번의 for 문을 실행. for문의 index = epoch
        model.train()
        loss_epoch = []
        metric_epoch = []

        '''
        train loop
        batch: (batch_size x C x H x W)
        0 ~ (전체 data 개수 / batch_size) 번의 for문 실행
        '''
        for index, batch in enumerate(train_dataloader): 
            x = batch['image'].to(args.device)
            y = batch['label'].to(args.device)

            # forward
            yhat = model(x)

            loss = loss_function(yhat, y)
            metric = BC_metric(y, yhat) # metric 계산

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            '''
            한 epoch 내의 loss, metric 값 추가
            '''
            loss_epoch.append(loss.detach().cpu().numpy())
            metric_epoch.append(metric.detach().cpu().numpy())

        # Tensorboard save
        writer_train.add_scalar('loss', np.mean(loss_epoch), epoch + 1)
        writer_train.add_scalar('accuracy', np.mean(metric_epoch), epoch + 1)

        train_loss.append(np.mean(loss_epoch)) # 1 epoch 마다 train loss 추가
        train_metric.append(np.mean(metric_epoch)) # 1 epoch 마다 train metric 추가

        '''
        validation loop
        torch.no_grad() : autograd 꺼서 gradient 계산 안함: 메모리 사용량 save
        model.eval() : model의 layer들을 evaluation mode로 설정: inference 시에 layer 작동 변화(dropout layer 끔, batchnorm 저장된 파라미터 사용)
        '''
        with torch.no_grad():
            model.eval()
            loss_epoch = []
            metric_epoch = []

            for index, data in enumerate(val_dataloader):
                x = data['image'].to(device)
                y = data['label'].to(device)

                yhat = model(x)

                loss = loss_function(yhat, y)
                metric = metric_function(y, yhat)

                loss_epoch.append(loss.detach().cpu().numpy())
                metri.append(metric_function(y, yhat))

        # Tensorboard save
        writer_val.add_scalar('loss', np.mean(loss_epoch), epoch + 1)
        writer_val.add_scalar('accuracy', np.mean(metric_epoch), epoch + 1)

        val_loss.append(np.mean(loss_epoch))
        val_metric.append(np.mean(metric_epoch))

        # model, optimizer save all epoch
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
       "./%s/model_epoch%d.pth.tar.gz" % (checkpoint_dir, epoch))

    writer_train.close()
    writer_val.close()

    return train_loss, train_metric, val_loss, val_metric