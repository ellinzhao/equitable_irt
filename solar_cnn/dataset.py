import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset

from utils import load_im
from utils import raw2temp


class SolarDataset(Dataset):

    def __init__(self, dataset_dir, ids, transform=None, train=False):
        self.dataset_dir = dataset_dir
        self.ids = ids
        self.transform = transform
        self.train = train
        self.loader = lambda path: load_im(path, raw2temp).astype(np.float32)

        dfs_and_ims = [self._load_sub_labels(sid) for sid in ids]
        labels, base_ims = list(zip(*dfs_and_ims))
        self.labels = pd.concat(labels)
        self.base_ims = {self.ids[i]: base_ims[i] for i in range(len(self.ids))}

    def _load_sub_labels(self, sid):
        base_df = pd.read_csv(os.path.join(self.dataset_dir, sid, 'base.csv'))
        cool_df = pd.read_csv(os.path.join(self.dataset_dir, sid, 'cool.csv'))

        # Remove data after 2 minutes of cooling
        cool_idx = cool_df['fname'].str.extract('(\d+)').iloc[:, 0]
        cool_idx = pd.to_numeric(cool_idx)
        cool_df = cool_df[cool_idx < 4 * 60 * 2]

        # Combine base and cool labels and remove NUC data
        df = pd.concat([base_df, cool_df])
        df = df[df['nuc_flag'] == 0]
        df['fname'] = df['fname'].apply(lambda x: os.path.join(self.dataset_dir, sid, x))

        # Load base ims and compute average
        base_idx = df['fname'].str.contains('base')
        base_avg = np.zeros((90, 60))
        for fname in df[base_idx]['fname']:
            base_avg += self.loader(fname)
        base_avg /= np.array(base_idx).sum()
        return df, base_avg

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname, delta = self.labels[['fname', 'delta']].iloc[idx]
        base_im = self.base_ims[os.path.dirname(fname)]
        image = self.loader(fname)

        if self.transform:
            image = self.transform(image)
            base_im = self.transform(base_im)

        # if self.train:
        #     # Add fever with probability of 0.3
        #     if np.random.uniform() < 0.5:
        #         image += np.random.uniform(1, 2)

        if 'base' in fname:
            delta = np.zeros(image.shape)
            delta = torch.tensor(delta)
        else:
            delta = image - base_im
        return image.float()[:, :, :], delta.float()[:, :, :]

    def display_im(img, label):
        plt.imshow(img.permute(1, 2, 0)[:, :, 0])

    def show_batch(dl):
        # Plot images grid of single batch
        for images, _ in dl:
            _, ax = plt.subplots(figsize=(16, 12))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            break
