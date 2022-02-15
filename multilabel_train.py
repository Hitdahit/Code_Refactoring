import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data_loader import *

GPU_NUM=0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device)

def multilabel_one_hot(labels, num_classes): # 로더에 추가 하면 좋을것 같아요
    # multilabel(class가 5개 일때) 
    # [0, 1, 3]
    # -> one_hot ecoding
    # -> multilabel(class가 5개 일때) 
    #-> [1,0,0,0,0]
    #   [0,1,0,0,0]
    #   [0,0,0,1,0]
    # 행 기준 더하기 -> [1,1,0,1,0]
    y_onehot = F.one_hot(labels, num_classes).sum(dim=0).float()  
    return y_onehot

def multilabel_classification_train(args):
    train_img_dir, train_label_dir, val_img_dir, val_label_dir, data_type, batch_size, workers, logs_dir,\
    epochs, model, loss_function, accuracy_function, optimizer, num_classes = _parser(args)
    
    train_dataloader = get_loader(train_img_dir, train_label_dir, 
                                  data_type, batch_size, workers)
    val_dataloader = get_loader(val_img_dir, val_label_dir, 
                                       data_type, batch_size, workers)

    writer_train = SummaryWriter(logs_dir=os.path.join(logs_dir, 'train'))
    writer_val = SummaryWriter(logs_dir=os.path.join(logs_dir, 'val'))
    
    train_loss = []
    train_accuarcy = []
    val_loss = []
    val_accuarcy = []

    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss_epoch = []
        train_accuracy_epoch = 0 # iter당 accuracy 계산값
        train_dataset_count = 0 # accuracy 계산을 위한 data 갯수 count

        for data in train_dataloader: # enumerate -> in
            x = data['image'].to(device)
            y = data['label'].to(device) # multilabel_one_hot_encoding

            yhat = model(x)

            loss = loss_function(yhat, y)
            
            pred = yhat.round().detach().cpu().numpy() # 0.5이상 1, 0.5 미만 0으로 바꿔줌, gpu->cpu->numpy (accuray 계산 할때 numpy로 했음)
            accuracy = accuracy_function(y.detach().cpu().numpy(), pred) #accuracy 계산 코드 전부 numpy로 들어감

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_epoch.append(loss.item())
            train_accuracy_epoch += accuracy # iter 당 accuracy 더하기로 변경
            train_dataset_count += x.shape[0] # accuracy 계산을 위한 data 갯수 count
        
        # Tensorboard save
        writer_train.add_scalar('loss', np.mean(train_loss_epoch), epoch + 1)
        writer_train.add_scalar('accuracy', train_accuracy_epoch, epoch + 1) #accuracy로 변경
        
        train_loss.append(np.mean(train_loss_epoch))
        train_accuarcy.append(train_accuracy_epoch / train_dataset_count) # 한 epoch 끝나면 accuracy 계산 
        #한 epoch당 train loss, train acc 출력
        print(f'Train Loss : {np.mean(train_loss_epoch)} Accuracy : {train_accuracy_epoch / train_dataset_count}')
        
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
                
                pred = yhat.round().detach().cpu().numpy() # 0.5이상 1, 0.5 미만 0으로 바꿔줌 gpu->cpu->numpy
                accuracy = accuracy_function(y.detach().cpu().numpy(), pred) #accuracy 계산 코드 전부 numpy로 들어감

                val_loss_epoch.append(loss.item())
                val_accuracy_epoch += accuracy # 한 iter 당 accuracy 더하기로 변경
                val_dataset_count += x.shape[0] # accuracy 계산을 위한 data 갯수 count
        
        # Tensorboard save
        writer_val.add_scalar('loss', np.mean(val_loss_epoch), epoch + 1)
        writer_val.add_scalar('accuracy', val_accuracy_epoch, epoch + 1) #accuracy로 변경

        val_loss.append(np.mean(val_loss_epoch))
        val_accuarcy.append(val_accuracy_epoch / val_dataset_count) # 한 epoch 끝나면 accuracy 계산 
        #한 epoch당 val loss, val acc 출력
        print(f'Val Loss : {np.mean(val_loss_epoch)} Accuaracy : {val_accuracy_epoch / val_dataset_count} ')
        
        # model, optimizer save all epoch
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
       "./%s/model_epoch%d.pth.tar.gz" % (checkpoint_dir, epoch))
        
    writer_train.close()
    writer_val.close()
    return train_loss, train_accuarcy, val_loss, val_accuarcy

'''
Example-Based Evaluation Metrics
'''

def emr(y_true, y_pred):
    '''
    모든 label에 대한 맞은 비율을 나타내는 방법
    모두 맞춰야되기 때문에 부분적으로 맞은 prediction을 무시한다는 단점이 있음.
    또한 그렇기 때문에 label에 대한 의존성이 고려된다.
    
    if y_true = np.array([[0,1,1], [1,1,1]])
       y_pred = np.array([[1,0,1], [0,0,0]])
    row_indicators = [False, False]
    exact_match_count = 0
    '''
    
    row_indicators = np.all(y_true == y_pred, axis = 1) # 각 라벨이 전부 일치해야 1, 하나라도 틀리면 0
    exact_match_count = np.sum(row_indicators) # 맞은 갯수 count
    return exact_match_count # accuracy 계산 하려면 data 갯수를 나눠줘야 됨 ex) acc = exact_match_count/len(dataset)


def example_based_accuracy(y_true, y_pred):
    '''
    일반적인 accuracy
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

"""
Label Based Metrics (macro, micro)
"""

def multilabel_metric_macro(y_true, y_pred):
    accuracy = label_based_macro_accuracy(y_true, y_pred)
    precision = label_based_macro_precision(y_true, y_pred)
    recall = label_based_macro_recall(y_true, y_pred)

    return accuracy, precision, recall

def mutlilabel_metric_micro(y_true, y_pred):
    accuracy = label_based_micro_accuracy(y_true, y_pred)
    precision = label_based_micro_precision(y_true, y_pred)
    recall = label_based_micro_recall(y_true, y_pred)

    return accuracy, precision, recall


def label_based_macro_accuracy(y_true, y_pred):

    # axis = 0 computes true positives along columns i.e labels
    l_acc_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)

    # axis = 0 computes true postive + false positive + false negatives along columns i.e labels
    l_acc_den = np.sum(np.logical_or(y_true, y_pred), axis = 0)

    # compute mean accuracy across labels. 
    return np.mean(l_acc_num/l_acc_den)

def label_based_macro_precision(y_true, y_pred):
	
	# axis = 0 computes true positive along columns i.e labels
	l_prec_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)

	# axis = computes true_positive + false positive along columns i.e labels
	l_prec_den = np.sum(y_pred, axis = 0)

	# compute precision per class/label
	l_prec_per_class = l_prec_num/l_prec_den

	# macro precision = average of precsion across labels. 
	l_prec = np.mean(l_prec_per_class)
	return l_prec

def label_based_macro_recall(y_true, y_pred):
    
    # compute true positive along axis = 0 i.e labels
    l_recall_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)

    # compute true positive + false negatives along axis = 0 i.e columns
    l_recall_den = np.sum(y_true, axis = 0)

    # compute recall per class/label
    l_recall_per_class = l_recall_num/l_recall_den

    # compute macro averaged recall i.e recall averaged across labels. 
    l_recall = np.mean(l_recall_per_class)
    return l_recall



def label_based_micro_accuracy(y_true, y_pred):
    
    # sum of all true positives across all examples and labels 
    l_acc_num = np.sum(np.logical_and(y_true, y_pred))

    # sum of all tp+fp+fn across all examples and labels.
    l_acc_den = np.sum(np.logical_or(y_true, y_pred))

    # compute mirco averaged accuracy
    return l_acc_num/l_acc_den


def label_based_micro_precision(y_true, y_pred):
    
    # compute sum of true positives (tp) across training examples
    # and labels. 
    l_prec_num = np.sum(np.logical_and(y_true, y_pred))

    # compute the sum of tp + fp across training examples and labels
    l_prec_den = np.sum(y_pred)

    # compute micro-averaged precision
    return l_prec_num/l_prec_den

def label_based_micro_recall(y_true, y_pred):
	
    # compute sum of true positives across training examples and labels.
    l_recall_num = np.sum(np.logical_and(y_true, y_pred))
    # compute sum of tp + fn across training examples and labels
    l_recall_den = np.sum(y_true)

    # compute mirco-average recall
    return l_recall_num/l_recall_den


