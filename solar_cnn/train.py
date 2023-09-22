import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import pickle

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from dataset import SolarDataset
from model import SolarRegression


# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
mean, std = (33.4878, 15.8042)  # C  TODO: recalculate this value


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def evaluate_test(model, test_dataset):
    mae = {
        'cool': 0,
        'base': 0,
        'cool_arr': [],
        'base_arr': [],
    }
    df_arr = []
    name_arr = []
    for i, (data, labels) in enumerate(test_dataset):
        data = data.to('cuda')
        labels = labels.to('cuda')

        fname = test_dataset.labels.iloc[i][1]
        key = fname.split('_')[-2]
        if key == 'cool':
            name_arr.append(fname.split('_')[2].split('/')[0])
        out = model(torch.unsqueeze(data, 0))
        mae_val = np.abs(labels.cpu() - out.cpu().detach().numpy()).sum()
        mae[key] += mae_val
        mae[f'{key}_arr'] += [(labels.item(), out.item())]

        df_arr += [(test_dataset.labels.iloc[i][1], labels.item(), out.item())]
    return mae


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr, weight_decay=1e-1)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def plot_accuracies(history):
    ''' Plot the history of accuracies'''
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')


def plot_losses(history):
    ''' Plot the losses in each epoch'''
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')


if __name__ == '__main__':
    # TODO: set this value
    dataset_dir = 'x'
    pid_folders = os.listdir(dataset_dir)
    fold_num = 0

    # TODO: for now, manually specify the train/val/test split from pid_folders.
    train_sids = []
    val_sids = []
    test_sids = []

    # INITIALIZATION ##########################
    # Define Model + Hyperparameters
    model = SolarRegression(save_str=str(fold_num)).to('cuda')
    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = 2e-5
    batch_size = 256

    # Datatset + Dataloaders
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize((64, 64), antialias=None),
    ])

    dataset = SolarDataset(dataset_dir, train_sids, transform=tform, train=True)
    val_dataset = SolarDataset(dataset_dir, val_sids, transform=tform)
    test_dataset = SolarDataset(dataset_dir, test_sids, transform=tform)

    train_dl = DataLoader(dataset, batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size)

    # TRAIN ####################################

    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    plot_losses(history)

    # TEST ####################################
    model = SolarRegression().to('cuda')
    model.load_state_dict(torch.load(f'./ckpts/model_params_{fold_num}.pth'))
    model.eval()

    # Evaluate Steady-State and Post-Solar Loading Data
    mae = evaluate_test(model, test_dataset)

    # print results
    for k in ['cool', 'base']:
        print(k, mae[k].item() / len(mae[f'{k}_arr']))

    # save the results in pickle
    with open(f'./results/results_{fold_num}.pkl', 'wb') as f:
        pickle.dump(mae, f)
