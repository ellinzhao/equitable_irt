import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from torch.utils.data import Dataset

from utils import load_im
from utils import raw2temp


class SolarDataset(Dataset):

    def __init__(self, labels, transform=None, train=False):
        self.labels = labels
        self.transform = transform
        self.train = train
        self.loader = lambda path: load_im(path, raw2temp).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname, y = self.labels.iloc[idx, 1:]
        frame_num = fname.split('_')[-1].split('.')[0]
        if int(frame_num) > 800 and self.train:
            return self.__getitem__(np.random.randint(0, len(self.labels)))

        image = self.loader(fname)
        y = np.array([y]).astype('float')

        if self.train:
            # add fever with probability of 0.3
            if np.random.uniform() < 0.3:
                image += np.random.uniform(0, 2)

        if self.transform:
            image = self.transform(image)
        return image.float()[:, :, :], torch.tensor(y, dtype=torch.float)

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
