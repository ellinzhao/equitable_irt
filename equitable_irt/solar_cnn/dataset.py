import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils import load_im
from ..utils import raw2temp

bgr2gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


class SolarDataset(Dataset):

    def __init__(self, dataset_dir, ids, num_load=60, transform=None, ir_transform=None, train=False):
        self.dataset_dir = dataset_dir
        self.ids = ids
        self.transform = transform
        self.ir_transform = ir_transform
        self.train = train
        self.num_load = num_load

        self.loader = lambda path: load_im(path, raw2temp).astype(np.float32)
        self.gray_loader = lambda path: load_im(path, bgr2gray).astype(np.float32)
        self.labels = pd.concat([self._load_sub_labels(sid) for sid in ids])

    def _load_sub_labels(self, sid):
        fix_dir = lambda x: os.path.join(self.dataset_dir, sid, x)
        df = pd.read_csv(fix_dir('label.csv')).iloc[:self.num_load]
        df['ir_fname'] = df['ir_fname'].apply(fix_dir)
        df['rgb_fname'] = df['rgb_fname'].apply(fix_dir)
        df['base_ir_fname'] = df['base_ir_fname'].apply(fix_dir)
        df['base_rgb_fname'] = df['base_rgb_fname'].apply(fix_dir)
        return df

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.labels.iloc[idx]
        # Load IR image
        ir_fname = row['ir_fname']
        ir = self.loader(ir_fname)
        base_ir_fname = row['base_ir_fname']
        base_ir = self.loader(base_ir_fname)
        session_type = [base_ir_fname == ir_fname, (base_ir_fname != ir_fname)]

        rgb_fname = row['rgb_fname']
        rgb_fname = row['rgb_fname'].replace('.png', '.jpg')  # TODO(ellin): fix this.
        gray = self.gray_loader(rgb_fname)

        if self.train and session_type[0] and random.uniform(0, 1) > 0.5:
            # For baseline images, add temp offset to mimic fever
            offset = random.uniform(2, 4)
            ir += offset
            base_ir += offset

        data_input = np.dstack([ir, gray])

        if self.transform:
            data_input = self.transform(data_input)
            base_ir = self.ir_transform(base_ir)
            session_type = torch.tensor(session_type)

        return data_input, base_ir, session_type
