import torch 
import torch.nn as nn # loss function, nn.Module
import torch.optim as optim # optimizer 사용

import os # os.path.join 사용
import numpy as np

from util import BC_metric, ML_metric, MC_metric
from sklearn.metrics import accuracy_score
from data_loader import get_loader


import wandb
from wandb import AlertLevel
from datetime import timedelta

'''
binary
'''
def binary_classification_train(args):
    '''
    binary classification training&validation 함수
    input: args
    output: train_loss, train_metric, val_loss, val_metric
    '''
    # data loading
    train_dataloader = get_loader(args, mode='train')
    val_dataloader = get_loader(args, mode='valid')

    # wandb init
    wandb.init(project= args.project_name,                 # 추가 해야됨 
               entity= args.team_name ,                    # 추가 해야됨
               config={"threshold": args.wadb_threshold,   # 추가 해야됨
                        "learning_rate": args.learning_rate ,
                        "epochs": args.epochs ,
                        "batch_size": args.batch_size  ,
                        })

    config = wandb.config
    wandb_percentage = args.wandb_percentage               # 추가 해야됨
    onceroof = []
    
    # epoch 마다 하나씩 추가될 list들
    train_loss = []
    train_metric = []
    val_loss = []
    val_metric = []

    # select whether you use Automatic Mixed Precision or not
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model = args.model
    loss_function = args.loss()
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
            with torch.cuda.amp.autocast(enabled=use_amp):
                yhat = model(x)
                loss = loss_function(yhat, y)
                
            metric = BC_metric(y, yhat, args.logit_thresh)
            
            # backward
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            '''
            한 epoch 내의 loss, metric 값 추가
            '''
            loss_epoch.append(loss.detach().cpu().numpy())
            metric_epoch += metric
            dataset_count += x.shape[0]
            
        train_loss.append(np.mean(loss_epoch)) # 1 epoch 마다 train loss 추가
        train_metric.append(metric_epoch / dataset_count) # 1 epoch 마다 train metric 추가
                
        
        # wandb 
        wandb.log({"Train loss": np.mean(train_loss), "Train Accuracy": np.mean(train_metric)}) # 메트릭을 추적.
        wandb.watch(model, criterion=loss_function, log='all', log_freq=10)  # wandb에서 학습할 모델 정보 추적

        if np.mean(train_metric) <= config.threshold:
            wandb.alert(
            title='Low Train Accuracy',
            text=f'Accuracy {np.mean(train_metric)} at step {epoch} is below the acceptable theshold',
            level=AlertLevel.WARN,
            wait_duration=timedelta(minutes=0)
            )
            # break
            
        if int(epoch/args.epochs*100) == wandb_percentage:
            if onceroof == []:
                print(epoch)
                wandb.alert(
                title= f'EPOCH',
                text= f'Step {epoch} "{wandb_percentage}%" done',
                level=AlertLevel.INFO,
                wait_duration=timedelta(minutes=0)
                )
                onceroof.append('not empty')

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
                x = data['image'].to(args.device)
                y = data['label'].to(args.device)

                yhat = model(x)

                loss = loss_function(yhat, y)
                metric = BC_metric(y, yhat, args.logit_thresh)

                loss_epoch.append(loss.detach().cpu().numpy())
                metric_epoch += metric
                dataset_count += x.shape[0]

        val_loss.append(np.mean(loss_epoch))
        val_metric.append(metric_epoch / dataset_count)
                
        # wandb 
        wandb.log({"Valid loss": np.mean(val_loss), "Valid Accuracy": np.mean(val_metric)}) # 메트릭을 추적.
        wandb.watch(model, criterion=loss_function, log='all', log_freq=10)
        
        if np.mean(val_metric) <= config.threshold:
            wandb.alert(
            title='Low Valid Accuracy',
            text=f'Accuracy {np.mean(val_metric)} at step {epoch} is below the acceptable theshold',
            level=AlertLevel.WARN,
            wait_duration=timedelta(minutes=0)
            )
            # break

        # model, optimizer save all epoch
        if use_amp:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler':  scaler.state_dict()},
           "./%s/model_epoch%d.pth.tar.gz" % (args.checkpoint_dir, epoch))
        else:        
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
           "./%s/model_epoch%d.pth.tar.gz" % (args.checkpoint_dir, epoch))

    return train_loss, train_metric, val_loss, val_metric


'''
multilabel
'''

def multilabel_classification_train(args):
    '''
    multilabel classification train&validation 함수
    input: args
    output: train_loss, train_metric, val_loss, val_metric
    '''
    #data loading
    train_dataloader = get_loader(args)
    val_dataloader = get_loader(args)

    # wandb init
    wandb.init(project= args.project_name,                 # 추가 해야됨 
               entity= args.team_name ,                    # 추가 해야됨
               config={"threshold": args.wadb_threshold,   # 추가 해야됨
                        "learning_rate": args.learning_rate ,
                        "epochs": args.epochs ,
                        "batch_size": args.batch_size  ,
                        })

    config = wandb.config
    wandb_percentage = args.wandb_percentage               # 추가 해야됨
    onceroof = []
    
    # epoch 마다 하나씩 추가될 list들
    train_loss = []
    train_metric = []
    val_loss = []
    val_metric = []

    # select whether you use Automatic Mixed Precision or not
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model = args.model
    loss_function = args.loss
    optimizer = args.optimizer
    model.to(args.device) # model이 GPU에서 돌아가도록 설정

    for epoch in range(args.epochs):
        model.train()
        loss_epoch = []
        metric_epoch = 0 # iter당 accuracy 계산값
        dataset_count = 0 # accuracy 계산을 위한 data 갯수 count

        '''
        train loop
        batch: (batch_size X C X H X W)
        0 ~ (전체 data 개수 / batch_size) 번의 for문 실행
        '''
        for index, data in enumerate(train_dataloader): 
            x = data['image'].to(args.device)
            y = data['label'].to(args.device) # multilabel_one_hot_encoding

            # forward
            with torch.cuda.amp.autocast(enabled=use_amp):
                yhat = model(x)
                loss = loss_function(yhat, y)

            pred = yhat.round().detach().cpu().numpy() # 0.5이상 1, 0.5 미만 0으로 바꿔줌, gpu->cpu->numpy (accuracy 계산 할때 numpy로 했음)
            metric = ML_metric(y.detach().cpu().numpy(), pred) #accuracy 계산 코드 전부 numpy로 들어감

            # backward
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            '''
            한 epoch 내의 loss, metric 값 추가
            '''
            loss_epoch.append(loss.item())
            metric_epoch += metric # iter 당 accuracy 더하기로 변경
            dataset_count += x.shape[0] # accuracy 계산을 위한 data 갯수 count
        
        train_loss.append(np.mean(loss_epoch))
        train_metric.append(metric_epoch / dataset_count) # 한 epoch 끝나면 accuracy 계산 
            
        # wandb 
        wandb.log({"Train loss": np.mean(train_loss), "Train Accuracy": np.mean(train_metric)}) # 메트릭을 추적.
        wandb.watch(model, criterion=loss_function, log='all', log_freq=10)  # wandb에서 학습할 모델 정보 추적

        if np.mean(train_metric) <= config.threshold:
            wandb.alert(
            title='Low Train Accuracy',
            text=f'Accuracy {np.mean(train_metric)} at step {epoch} is below the acceptable theshold',
            level=AlertLevel.WARN,
            wait_duration=timedelta(minutes=0)
            )
            # break
            
        if int(epoch/args.epochs*100) == wandb_percentage:
            if onceroof == []:
                print(epoch)
                wandb.alert(
                title= f'EPOCH',
                text= f'Step {epoch} "{wandb_percentage}%" done',
                level=AlertLevel.INFO,
                wait_duration=timedelta(minutes=0)
                )
                onceroof.append('not empty')
    
        '''
        validation loop
        torch.no_grad() : autograd 꺼서 gradient 계산 안함: 메모리 사용량 save
        model.eval() : model의 layer들을 evaluation mode로 설정: inference 시에 layer 작동변화(dropout layer 끔, batchnorm 저장된 파라미터 사용)
        '''
        with torch.no_grad():
            model.eval()
            loss_epoch = []
            metric_epoch = 0 # iter당 accuracy 계산값
            dataset_count = 0 # accuracy 계산을 위한 data 갯수 count

            for index, data in enumerate(val_dataloader): 
                # forward pass
                x = data['image'].to(args.device)
                y = data['label'].to(args.device) # multilabel_one_hot_encoding

                yhat = model(x)

                loss = loss_function(yhat, y)
                
                pred = yhat.round().detach().cpu().numpy() # 0.5이상 1, 0.5 미만 0으로 바꿔줌 gpu->cpu->numpy
                metric = ML_metric(y.detach().cpu().numpy(), pred) #accuracy 계산 코드 전부 numpy로 들어감

                loss_epoch.append(loss.item())
                metric_epoch += metric # 한 iter 당 accuracy 더하기로 변경
                dataset_count += x.shape[0] # accuracy 계산을 위한 data 갯수 count
        
        val_loss.append(np.mean(loss_epoch))
        val_metric.append(metric_epoch / dataset_count) # 한 epoch 끝나면 accuracy 계산 
        
        # wandb 
        wandb.log({"Valid loss": np.mean(val_loss), "Valid Accuracy": np.mean(val_metric)}) # 메트릭을 추적.
        wandb.watch(model, criterion=loss_function, log='all', log_freq=10)
        
        if np.mean(val_metric) <= config.threshold:
            wandb.alert(
            title='Low Valid Accuracy',
            text=f'Accuracy {np.mean(val_metric)} at step {epoch} is below the acceptable theshold',
            level=AlertLevel.WARN,
            wait_duration=timedelta(minutes=0)
            )
            # break
        
        # model, optimizer save all epoch
        if use_amp:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler':  scaler.state_dict()},
           "./%s/model_epoch%d.pth.tar.gz" % (args.checkpoint_dir, epoch))
        else:        
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
           "./%s/model_epoch%d.pth.tar.gz" % (args.checkpoint_dir, epoch))


    return train_loss, train_metric, val_loss, val_metric


'''
multiclass
'''
def multiclass_classification_train(args):
    '''
    multiclass classification training&validation 함수
    input: args
    output: train_loss, train_metric, val_loss, val_metric
    '''
    # data loading
    train_dataloader = get_loader(args, mode='train')
    val_dataloader = get_loader(args, mode='valid')

    # wandb init
    wandb.init(project= args.project_name,                 # 추가 해야됨 
               entity= args.team_name ,                    # 추가 해야됨
               config={"threshold": args.wadb_threshold,   # 추가 해야됨
                        "learning_rate": args.learning_rate ,
                        "epochs": args.epochs ,
                        "batch_size": args.batch_size  ,
                        })

    config = wandb.config
    wandb_percentage = args.wandb_percentage               # 추가 해야됨
    onceroof = []
    
    # epoch 마다 하나씩 추가될 list들
    train_loss = []
    train_metric = []
    val_loss = []
    val_metric = []

    # select whether you use Automatic Mixed Precision or not
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
            x = data['image'].to(args.device)
            y = data['label'].to(args.device) 

            # forward
            with torch.cuda.amp.autocast(enabled=use_amp):
                yhat = model(x)
                loss = loss_function(yhat, y)
            metric = MC_metric(y, yhat)

            # backward
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_epoch.append(loss.detach().cpu().numpy())
            metric_epoch += metric
            dataset_count += x.shape[0]

        # scheduler.step()

        train_loss.append(np.mean(loss_epoch)) # 1 epoch 마다 train loss 추가
        train_metric.append(metric_epoch / dataset_count) # 1 epoch 마다 train metric 추가
        
        # wandb 
        wandb.log({"Train loss": np.mean(train_loss), "Train Accuracy": np.mean(train_metric)}) # 메트릭을 추적.
        wandb.watch(model, criterion=loss_function, log='all', log_freq=10)  # wandb에서 학습할 모델 정보 추적

        if np.mean(train_metric) <= config.threshold:
            wandb.alert(
            title='Low Train Accuracy',
            text=f'Accuracy {np.mean(train_metric)} at step {epoch} is below the acceptable theshold',
            level=AlertLevel.WARN,
            wait_duration=timedelta(minutes=0)
            )
            # break
            
        if int(epoch/args.epochs*100) == wandb_percentage:
            if onceroof == []:
                print(epoch)
                wandb.alert(
                title= f'EPOCH',
                text= f'Step {epoch} "{wandb_percentage}%" done',
                level=AlertLevel.INFO,
                wait_duration=timedelta(minutes=0)
                )
                onceroof.append('not empty')        
        
        # validation
        with torch.no_grad():
            model.eval()
            val_loss_epoch = []
            val_accuracy_epoch = 0 # iter당 accuracy 계산값
            val_dataset_count = 0 # accuracy 계산을 위한 data 갯수 count

            for data in val_dataloader: # enumerate -> in
                # forward pass
                x = data['image'].to(args.device)
                y = data['label'].to(args.device) # multilabel_one_hot_encoding

                yhat = model(x)

                loss = loss_function(yhat, y)
                metric = MC_metric(y, yhat)

                loss_epoch.append(loss.detach().cpu().numpy())
                metric_epoch += metric
                dataset_count += x.shape[0]
        
        val_loss.append(np.mean(loss_epoch))
        val_metric.append(metric_epoch / dataset_count)
        
        # wandb 
        wandb.log({"Valid loss": np.mean(val_loss), "Valid Accuracy": np.mean(val_metric)}) # 메트릭을 추적.
        wandb.watch(model, criterion=loss_function, log='all', log_freq=10)
        
        if np.mean(val_metric) <= config.threshold:
            wandb.alert(
            title='Low Valid Accuracy',
            text=f'Accuracy {np.mean(val_metric)} at step {epoch} is below the acceptable theshold',
            level=AlertLevel.WARN,
            wait_duration=timedelta(minutes=0)
            )
            # break
        
        # model, optimizer save all epoch
        if use_amp:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler':  scaler.state_dict()},
           "./%s/model_epoch%d.pth.tar.gz" % (args.checkpoint_dir, epoch))
        else:        
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
           "./%s/model_epoch%d.pth.tar.gz" % (args.checkpoint_dir, epoch))

    return train_loss, train_metric, val_loss, val_metric