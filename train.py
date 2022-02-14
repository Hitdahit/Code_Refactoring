import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np

# val = validation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
TOBE
'''

def _parser(args):

    if 'binary' in args.task_type:
        n_classes = 2
    # loss path
    if 'cross' in args.loss:
        loss_function = nn.CrossEntropyLoss()
    # 
    if 'resnet50' in args.model:
        model = model.ResNet50_Classifier(n_classes=n_classes)
        
    if 'Adam' in args.optimizer:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        

    
    
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
    train_img_dir, train_label_dir, val_img_dir, val_label_dir, data_type, batch_size, workers, logs_dir,\
    epochs, model, loss, metric, optimizer = _parser(args)
    
    train_dataloader = get_loader(train_img_dir, train_label_dir, 
                                  data_type, batch_size, workers)
    val_dataloader = get_loader(val_image_dir, val_label_dir, 
                                       data_type, batch_size, workers)
    

    writer_train = SummaryWriter(logs_dir=os.path.join(logs_dir, 'train'))
    writer_val = SummaryWriter(logs_dir=os.path.join(logs_dir, 'val'))

    train_loss = []
    train_metric = []
    val_loss = []
    val_metric = []
    
    model.to(device)

    for epoch in range(epochs):
        model.train()
        loss_epoch = []
        metric_epoch = []

        # train
        for index, batch in enumerate(train_dataloader):
            x = batch['image'].to(device)
            y = batch['label'].to(device)

            yhat = model(x)

            loss = loss_function(yhat, y)
            metric = metric_function(y, yhat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.detach().cpu().numpy()
            metric_epoch += metric.detach().cpu().numpy()

        # Tensorboard save
        writer_train.add_scalar('loss', np.mean(loss_epoch), epoch + 1)
        writer_train.add_scalar('accuracy', np.mean(metric_epoch), epoch + 1) # metric? accuracy?

        train_loss.append(np.mean(loss_epoch))
        train_metric.append(np.mean(metric_epoch))

        # validation
        with torch.no_grad():
            model.eval()
            loss_epoch = []
            metric_epoch = []

            for index, data in enumerate(val_dataloader, 1):
                # forward pass
                x = data['image'].to(device)
                y = data['label'].to(device)

                yhat = model(x)

                loss = loss_function(yhat, y)
                metric = metric_function(y, yhat)

                loss_epoch += loss.detach().cpu().numpy()
                metric = metric_function(y, yhat)

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

    return train_loss, val_loss