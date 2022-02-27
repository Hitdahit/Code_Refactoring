import torch 
import torch.nn as nn # loss function, nn.Module
import torch.optim as optim # optimizer 사용
from torch.utils.tensorboard import SummaryWriter # 텐서보드 사용

import os # os.path.join 사용
import numpy as np

from sklearn.metrics import accuracy_score


'''
binary
BC_metric util.py로 가는 것은?
'''
def BC_metric(y, yhat):
    '''
    input
        y: B x 1
        yhat: B x 2
    output: 1 batch 중에서 맞춘 개수
    '''
    yhat = argmax(yhat, dim=1)
    metric = accuracy_score(y, yhat)
    
    return metric * len(y)

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
        metric_epoch = 0
        dataset_count = 0 # accuracy 계산을 위한 data 갯수 count

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
            metric = BC_metric(y, yhat)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            '''
            한 epoch 내의 loss, metric 값 추가
            '''
            loss_epoch.append(loss.detach().cpu().numpy())
            metric_epoch += metric
            dataset_count += x.shape[0]

        # Tensorboard save
        writer_train.add_scalar('loss', np.mean(loss_epoch), epoch + 1)
        writer_train.add_scalar('accuracy', metic_epoch / dataset_count, epoch + 1)

        train_loss.append(np.mean(loss_epoch)) # 1 epoch 마다 train loss 추가
        train_metric.append(metic_epoch / dataset_count) # 1 epoch 마다 train metric 추가

        '''
        validation loop
        torch.no_grad() : autograd 꺼서 gradient 계산 안함: 메모리 사용량 save
        model.eval() : model의 layer들을 evaluation mode로 설정: inference 시에 layer 작동 변화(dropout layer 끔, batchnorm 저장된 파라미터 사용)
        '''
        with torch.no_grad():
            model.eval()
            loss_epoch = []
            metric_epoch = 0
            dataset_count = 0

            for index, data in enumerate(val_dataloader):
                x = data['image'].to(device)
                y = data['label'].to(device)

                yhat = model(x)

                loss = loss_function(yhat, y)
                metric = BC_metric(y, yhat)

                loss_epoch.append(loss.detach().cpu().numpy())
                metric_epoch += metric
                dataset_count += x.shape[0]

        # Tensorboard save
        writer_val.add_scalar('loss', np.mean(loss_epoch), epoch + 1)
        writer_val.add_scalar('accuracy', metic_epoch / dataset_count, epoch + 1)

        val_loss.append(np.mean(loss_epoch))
        val_metric.append(metric_epoch / dataset_count)

        # model, optimizer save all epoch
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
       "./%s/model_epoch%d.pth.tar.gz" % (args.point_dir, epoch))

    writer_train.close()
    writer_val.close()

    return train_loss, train_metric, val_loss, val_metric


'''
multilabel
'''
def ML_metric(y_true, y_pred):
    '''
    일반적인 accuracy (example_based_accuracy)
    전체 predicted와 actual label 중에 맞은 predicted label 비율
    
    if y_true = np.array([[0,1,1], [1,1,1]])
       y_pred = np.array([[1,0,1], [0,0,0]])
    numerator = [1,0]
    denominator = [3,3]
    instance_accuracy = [0.333333, 0]
    np.sum(instance_accuracy) : 0.33333
    '''

    # compute true positive using the logical AND operator
    numerator = np.sum(np.logical_and(y_true, y_pred), axis = 1) 

    # compute true_positive + false negatives + false positive using the logical OR operator
    denominator = np.sum(np.logical_or(y_true, y_pred), axis = 1)
    instance_accuracy = numerator/denominator

    return np.sum(instance_accuracy) # accuracy 계산 하려면 data 갯수를 나눠줘야 됨

def multilabel_classification_train(args):
    '''
    multilabel classification train&validation 함수
    input: args
    output: train_loss, train_metric, val_loss, val_metric
    '''
    #data loading
    train_dataloader = get_loader(args)
    val_dataloader = get_loader(args)

    # 텐서보드를 사용하기 위한 SummaryWriter 설정, log 파일을 저장할 경로
    writer_train = SummaryWriter(logs_dir=os.path.join(args.log_dir, 'train'))
    writer_val = SummaryWriter(logs_dir=os.path.join(args.log_dir, 'val'))
    
    # epoch 마다 하나씩 추가될 list들
    train_loss = []
    train_accuarcy = []
    val_loss = []
    val_accuarcy = []

    model = args.model
    loss_function = args.loss
    optimizer = args.optimizer
    model.to(args.device) # model이 GPU에서 돌아가도록 설정

    for epoch in range(args.epochs):
        model.train()
        loss_epoch = []
        accuracy_epoch = 0 # iter당 accuracy 계산값
        dataset_count = 0 # accuracy 계산을 위한 data 갯수 count

        '''
        train loop
        batch: (batch_size X C X H X W)
        0 ~ (전체 data 개수 / batch_size) 번의 for문 실행
        '''
        for data in train_dataloader: 
            x = data['image'].to(args.device)
            y = data['label'].to(args.device) # multilabel_one_hot_encoding

            # forward
            yhat = model(x)

            loss = loss_function(yhat, y)
            
            pred = yhat.round().detach().cpu().numpy() # 0.5이상 1, 0.5 미만 0으로 바꿔줌, gpu->cpu->numpy (accuracy 계산 할때 numpy로 했음)
            accuracy = ML_metric(y.detach().cpu().numpy(), pred) #accuracy 계산 코드 전부 numpy로 들어감

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            '''
            한 epoch 내의 loss, metric 값 추가
            '''
            loss_epoch.append(loss.item())
            accuracy_epoch += accuracy # iter 당 accuracy 더하기로 변경
            dataset_count += x.shape[0] # accuracy 계산을 위한 data 갯수 count
        
        # Tensorboard save
        writer_train.add_scalar('loss', np.mean(loss_epoch), epoch + 1) # 1 epoch 마다 train loss 추가
        writer_train.add_scalar('accuracy', accuracy_epoch / dataset_count, epoch + 1) # 1 epoch 마다 train accuracy 추가
        
        train_loss.append(np.mean(loss_epoch))
        train_accuarcy.append(accuracy_epoch / dataset_count) # 한 epoch 끝나면 accuracy 계산 
    
        '''
        validation loop
        torch.no_grad() : autograd 꺼서 gradient 계산 안함: 메모리 사용량 save
        model.eval() : model의 layer들을 evaluation mode로 설정: inference 시에 layer 작동변화(dropout layer 끔, batchnorm 저장된 파라미터 사용)
        '''
        with torch.no_grad():
            model.eval()
            loss_epoch = []
            accuracy_epoch = 0 # iter당 accuracy 계산값
            dataset_count = 0 # accuracy 계산을 위한 data 갯수 count

            for data in val_dataloader: 
                # forward pass
                x = data['image'].to(args.device)
                y = data['label'].to(args.device) # multilabel_one_hot_encoding

                yhat = model(x)

                loss = loss_function(yhat, y)
                
                pred = yhat.round().detach().cpu().numpy() # 0.5이상 1, 0.5 미만 0으로 바꿔줌 gpu->cpu->numpy
                accuracy = ML_metric(y.detach().cpu().numpy(), pred) #accuracy 계산 코드 전부 numpy로 들어감

                loss_epoch.append(loss.item())
                accuracy_epoch += accuracy # 한 iter 당 accuracy 더하기로 변경
                dataset_count += x.shape[0] # accuracy 계산을 위한 data 갯수 count
        
        # Tensorboard save
        writer_val.add_scalar('loss', np.mean(loss_epoch), epoch + 1)
        writer_val.add_scalar('accuracy', accuracy_epoch / dataset_count, epoch + 1) #accuracy로 변경

        val_loss.append(np.mean(loss_epoch))
        val_accuarcy.append(accuracy_epoch / dataset_count) # 한 epoch 끝나면 accuracy 계산 
        
        
        # model, optimizer save all epoch
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
       "./%s/model_epoch%d.pth.tar.gz" % (args.checkpoint_dir, epoch))
        
    writer_train.close()
    writer_val.close()

    return train_loss, train_accuarcy, val_loss, val_accuarcy