import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data_loader import *

GPU_NUM=0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device)

def multiclass_classification_train(args):
    '''
    binary classification training&validation 함수
    input: args
    output: train_loss, train_metric, val_loss, val_metric
    '''
    # _parser 함수의 리턴값으로 사용할 파라미터 가져오기
    # train_img_dir: training에 사용할 image의 path
    train_img_dir, train_label_dir, val_img_dir, val_label_dir, data_type, batch_size, workers, logs_dir,\
    epochs, model, loss, metric, optimizer = _parser(args)
    
    # train에 사용할 data loading. batch_size별로 끊어서 출력됨
    train_dataloader = get_loader(train_img_dir, train_label_dir, 
                                  data_type, batch_size, workers)
    # validation에 사용할 data loading. batch_size별로 끊어서 출력됨
    val_dataloader = get_loader(val_image_dir, val_label_dir, 
                                       data_type, batch_size, workers)
    
    # 텐서보드를 사용하기 위한 SummaryWriter 설정, log 파일을 저장할 경로
    writer_train = SummaryWriter(logs_dir=os.path.join(logs_dir, 'train'))
    writer_val = SummaryWriter(logs_dir=os.path.join(logs_dir, 'val'))

    train_loss = [] # train loop의 loss를 저장하기 위한 빈 list. epoch마다 하나씩 추가
    train_metric = [] # train loop의 metirc을 저장하기 위한 빈 list. epoch마다 하나씩 추가
    val_loss = [] # validation loop의 loss를 저장하기 위한 빈 list. epoch마다 하나씩 추가
    val_metric = [] # validation loop의 metric를 저장하기 위한 빈 list. epoch마다 하나씩 추가
    
    model.to(device) # model이 GPU에서 돌아가도록 설정
    for epoch in range(epochs):
        model.train()
        train_loss_epoch = []
        train_accuracy_epoch = 0 # iter당 accuracy 계산값
        train_dataset_count = 0 # accuracy 계산을 위한 data 갯수 count

        train_acc_targets = []
        train_acc_outputs = []

        for i, data in train_dataloader: # enumerate -> in
            x = data['image'].to(device)
            y = data['label'].to(device) 

            yhat = model(x)
            loss = loss_function(yhat, y)

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

            train_loss_epoch.append(loss.item())
            train_accuracy_epoch += accuracy # iter 당 accuracy 더하기로 변경
            train_dataset_count += x.shape[0] # accuracy 계산을 위한 data 갯수 count
        
        scheduler.step()

        # Tensorboard save
        writer_train.add_scalar('loss', np.mean(train_loss_epoch), epoch + 1)
        writer_train.add_scalar('accuracy', train_accuracy_epoch, epoch + 1) #accuracy로 변경
        
        train_loss.append(np.mean(train_loss_epoch))
        train_accuarcy.append(train_accuracy_epoch / train_dataset_count) # 한 epoch 끝나면 accuracy 계산 
        #한 epoch당 train loss, train acc 출력
        print(f'Epoch : {epoch} Train Loss : {np.mean(train_loss_epoch)} Accuracy : {train_accuracy_epoch / train_dataset_count}')
        
        # validation
        with torch.no_grad():
            model.eval()
            val_loss_epoch = []
            val_accuracy_epoch = 0 # iter당 accuracy 계산값
            val_dataset_count = 0 # accuracy 계산을 위한 data 갯수 count

            val_acc_targets = []
            val_acc_outputs = []

            for data in val_dataloader: # enumerate -> in
                # forward pass
                x = data['image'].to(device)
                y = data['label'].to(device) # multilabel_one_hot_encoding

                yhat = model(x)

                loss = loss_function(yhat, y)

                y_temp = y
                for t in y_temp.view(-1,1).cpu():
                    val_acc_targets.append(t.item()) 

                _, yhat_temp = torch.max(yhat, 1)
                for u in yhat_temp.view(-1,1).cpu():
                    val_acc_outputs.append(u.item())

                accuracy = accuracy_function(val_acc_outputs, val_acc_targets)

                val_loss_epoch.append(loss.item())
                val_accuracy_epoch += accuracy # 한 iter 당 accuracy 더하기로 변경
                val_dataset_count += x.shape[0] # accuracy 계산을 위한 data 갯수 count
        
        # Tensorboard save
        writer_val.add_scalar('loss', np.mean(val_loss_epoch), epoch + 1)
        writer_val.add_scalar('accuracy', val_accuracy_epoch, epoch + 1) #accuracy로 변경

        val_loss.append(np.mean(val_loss_epoch))
        val_accuarcy.append(val_accuracy_epoch / val_dataset_count) # 한 epoch 끝나면 accuracy 계산 
        #한 epoch당 val loss, val acc 출력
        print(f'Epoch : {epoch} Val Loss : {np.mean(val_loss_epoch)} Accuaracy : {val_accuracy_epoch / val_dataset_count} ')
        
        # model, optimizer save all epoch
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
       "./%s/model_epoch%d.pth.tar.gz" % (checkpoint_dir, epoch))
        
    writer_train.close()
    writer_val.close()
    return train_loss, train_accuarcy, val_loss, val_accuarcy
