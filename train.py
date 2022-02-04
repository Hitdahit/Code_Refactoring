import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def binary_classification_train(model, loss_function, metric_function, optimizer, epochs):
    train_dataloader = get_loader(train_image_dir, train_label_dir, data_type, batch_size, workers)
    validation_dataloader = get_loader(validation_image_dir, validation_label_dir, data_type, batch_size, workers)

    writer_train = SummaryWriter(logs_dir=os.path.join(logs_dir, 'train'))
    writer_val = SummaryWriter(logs_dir=os.path.join(logs_dir, 'val'))

    train_loss = []
    validation_loss = []

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

            loss_epoch.append(loss.item())
            metric_epoch.append(metric)

        # Tensorboard save
        writer_train.add_scalar('loss', np.mean(loss_epoch), epoch + 1)

        train_loss.append(np.mean(loss_epoch))

        # validation
        with torch.no_grad():
            model.eval()
            loss_epoch = []
            metric_epoch = []

            for index, data in enumerate(validation_dataloader, 1):
                # forward pass
                x = data['image'].to(device)
                y = data['label'].to(device)

                yhat = model(x)

                loss = loss_function(yhat, y)
                metric = metric_function(y, yhat)

                loss_epoch.append(loss.item())
                metric_epoch.append(metric)

            # Tensorboard save
        writer_val.add_scalar('loss', np.mean(loss_epoch), epoch + 1)

        validation_loss.append(np.mean(loss_epoch))

        # model, optimizer save all epoch
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
       "./%s/model_epoch%d.pth.tar.gz" % (checkpoint_dir, epoch))

    writer_train.close()
    writer_val.close()

    return train_loss, validation_loss