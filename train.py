import torch 
import torch.nn as nn # loss function, nn.Module
import torch.optim as optim # optimizer 사용
from torch.utils.tensorboard import SummaryWriter # 텐서보드 사용

import os # os.path.join 사용
import numpy as np

# val = validation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CPU/GPU에서 돌아갈지 정하는 device flag

'''
TOBE
'''


def _parser(args):

    task_type = args.task_type
    n_classes = int(args.num_class)
    
    losses = {'BCE':nn.BCEWithLogitsLoss(),'CE':nn.CrossEntropyLoss(), 'F':}  # 추후 추가
    loss = None
    for key, value in losses.items():
        if key in args.loss:
            loss = value
    
    models = {'vgg':model.VGG_Classifier(n_classes, int(args.model_size)),\
            'resnet':model.ResNet_Classifier(n_classes, int(args.model_size)),\
            'densenet':model.DenseNet_Classifier(n_classes, int(args.model_size)),\
            'efficientnet': model.EfficientNet_Classifier(n_classes, int(args.model_size))}
    model = None
    for key, value in models.items():
        if key in args.model:
            model = value

    
        
    

    
    
    train_image_dir = args.train_img_dir
    train_label_dir = args.train_label_dir
    val_img_dir = args.val_img_dir
    val_label_dir = args.val_label_dir
    
    epochs = args.epochs
    
    
    return loss, model, ...
 
 


'''
present
'''

def binary_classification_train(args):
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

    for epoch in range(epochs): # 0 ~ (epochs-1) 번의 for 문을 실행. for문의 index = epoch
        model.train() # 모델을 학습 모드로 변환
        loss_epoch = [] # 한 epoch 내에서 batch마다 계산될 loss 저장하는 빈 list
        metric_epoch = [] # 한 epoch 내에서 batch마다 계산될 metric 저장하는 빈 list

        # train loop
        # enumerate: index와 내용 같이 retrun
        # batch: (batch_size x C x H x W)
        # 0 ~ (전체 data 개수 / batch_size) 번의 for문 실행
        for index, batch in enumerate(train_dataloader): 
            x = batch['image'].to(device) # 배치의 image를 설정한 device(GPU)로 보냄
            y = batch['label'].to(device) # 배치의 label을 설정한 device(GPU)로 보냄

            yhat = model(x) # forward: model에 image(x)를 넣어서 가설(yhat) 획득

            loss = loss_function(yhat, y) # 가설(yhat)과 ground truth(y) 비교해 loss 계산
            metric = metric_function(y, yhat) # metric 계산

            optimizer.zero_grad() # gradient 초기화
            loss.backward() # 구한 loss로부터 back propagation을 통해 각 변수마다 loss에 대한 gradient 를 구해주기
            optimizer.step() # 계산한 기울기 + opimizer의 알고리즘에 맞춰 model의 파라미터 업데이트

            loss_epoch.append(loss.detach().cpu().numpy()) # 한 epoch 내의 loss 값 추가
            metric_epoch.append(metric.detach().cpu().numpy()) # 한 epoch 내의 metric 값 추가

        # Tensorboard save
        writer_train.add_scalar('loss', np.mean(loss_epoch), epoch + 1)
        writer_train.add_scalar('accuracy', np.mean(metric_epoch), epoch + 1) # metric? accuracy?

        train_loss.append(np.mean(loss_epoch)) # 1 epoch 마다 train loss 추가
        train_metric.append(np.mean(metric_epoch)) # 1 epoch 마다 train metric 추가

        # validation loop
        with torch.no_grad(): # autograd 꺼서 gradient 계산 안함: 메모리 사용량 save
            model.eval() # model의 layer들을 evaluation mode로 설정: inference 시에 layer 작동 변화(dropout layer 끔, batchnorm 저장된 파라미터 사용)
            loss_epoch = []
            metric_epoch = []

            for index, data in enumerate(val_dataloader):
                # forward pass
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