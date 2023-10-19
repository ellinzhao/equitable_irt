import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.best_val_loss = np.inf
        self.save_str = ''

    def training_step(self, batch):
        images, labels = batch
        images = images.to('cuda')
        labels = labels.to('cuda')
        out = self(images)                   # Generate predictions
        loss1 = F.mse_loss(out, labels)       # Calculate loss
        return loss1

    def validation_step(self, batch):
        images, labels = batch
        images = images.to('cuda')
        labels = labels.to('cuda')
        out = self(images)                      # Generate predictions
        loss = F.mse_loss(out, labels)
        return {'val_loss': loss.detach()}      # 'val_acc': acc

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        if epoch_loss.item() < self.best_val_loss and epoch_loss.item() < 2:
            self.best_val_loss = epoch_loss.item()
            print('Saving model...')
            torch.save(self.state_dict(), f'./ckpts/model_params_{self.save_str}.pth')
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print('Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}'.format(
            epoch, result['train_loss'], result['val_loss']))


class SolarRegression(RegressionBase):
    def __init__(self, save_str=''):
        super().__init__()
        self.save_str = save_str
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(16 * 8 * 8, 200)
        self.fc2 = nn.Linear(200, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = (F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = (F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = (F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x_embed = F.relu(self.fc2(x))
        x = self.fc3(x_embed)
        return x
