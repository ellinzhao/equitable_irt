import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Normalize
from torch.utils.data import Dataset

from ..utils import load_im
from ..utils import raw2temp

bgr2gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


class SolarDataset(Dataset):

    def __init__(self, dataset_dir, ids, num_load=60, transform=None,
                 train=False, fever_prob=0.5, session_filter=None):
        self.dataset_dir = dataset_dir
        self.ids = ids
        self.transform = transform
        self.train = train
        self.num_load = num_load
        self.fever_prob = fever_prob
        self.session_filter = session_filter

        # Mean and std for IR data
        self.mean, self.std = 0, 1
        if transform:
            for operation in transform.transforms:
                if type(operation) is Normalize:
                    self.mean = operation.mean[0]
                    self.std = operation.std[0]
                    break

        self.loader = lambda path: load_im(path, raw2temp).astype(np.float32)
        self.gray_loader = lambda path: load_im(path)[..., 2].astype(np.float32)
        self.labels = pd.concat([self._load_sub_labels(sid) for sid in ids])

    def _fname_index(self, x):
        session, i = re.findall(r'([a-z]+)_ir(\d+).png', os.path.basename(x))[0]
        return f'{session}_{i}'

    def _load_sub_labels(self, sid):
        # TODO(ellin): clean up this code
        sid_dir = lambda x: os.path.join(self.dataset_dir, sid, x)

        # The labels csv has the paired file names
        df = pd.read_csv(sid_dir('label.csv')).iloc[:self.num_load]
        df['ir_fname'] = df['ir_fname'].apply(sid_dir)
        df['rgb_fname'] = df['rgb_fname'].apply(sid_dir)
        df['base_ir_fname'] = df['base_ir_fname'].apply(sid_dir)
        df['base_rgb_fname'] = df['base_rgb_fname'].apply(sid_dir)
        df['session_i'] = df['ir_fname'].apply(self._fname_index)
        df['session_i'] = sid + '_' + df['session_i']
        df.set_index('session_i', inplace=True)

        # Load ROI csvs to get forehead and background temps
        base_df = pd.read_csv(sid_dir('base_temps.csv'))
        base_df['session_i'] = sid + '_base_' + base_df.iloc[:, 0].apply(str)
        base_df.set_index('session_i', inplace=True)
        cool_df = pd.read_csv(sid_dir('cool_temps.csv'))
        cool_df['session_i'] = sid + '_cool_' + cool_df.iloc[:, 0].apply(str)
        cool_df.set_index('session_i', inplace=True)
        roi_df = pd.concat([cool_df, base_df])[['bg', 'forehead', 'ymin', 'ymax', 'xmin', 'xmax']]

        df = pd.merge(df, roi_df, how='left', left_index=True, right_index=True)
        base_index = sid + '_' + df['base_ir_fname'].apply(self._fname_index)
        df['base_forehead'] = df['forehead'].loc[base_index.values].values
        df['base_bg'] = df['bg'].loc[base_index.values].values

        # There are more SL images than baseline images so duplicate the baseline rows for more data
        # Base images are randomly turned to fever images, so this ensures there is enough base data
        if self.train:
            mask = df.index.str.contains('_base_')
            base_df = df[mask]
            base_df.index = base_df.index + '_0'
            df = pd.concat([df, base_df])
        if self.session_filter:
            mask = df.index.str.contains(f'_{self.session_filter}_')
            df = df[mask]
        return df

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.labels.iloc[idx]

        bg = row['bg']
        forehead = row['forehead']
        base_forehead = row['base_forehead']
        ymin = row['ymin']
        ymax = row['ymax']
        xmin = row['xmin']
        xmax = row['xmax']

        # Load IR image
        ir_fname = row['ir_fname']
        ir = self.loader(ir_fname)
        base_ir_fname = row['base_ir_fname']
        base_ir = self.loader(base_ir_fname)
        session_type = [base_ir_fname == ir_fname, (base_ir_fname != ir_fname)]

        rgb_fname = row['rgb_fname']
        gray = self.gray_loader(rgb_fname)

        if self.train and session_type[0] and random.uniform(0, 1) < self.fever_prob:
            # For baseline images, add temp offset to mimic fever
            offset = random.uniform(2, 4)
            ir += offset
            base_ir += offset

        data_input = np.dstack([ir, base_ir, gray])

        if self.transform:
            data_input = self.transform(data_input)
            session_type = torch.tensor(session_type)
            bg = (bg - self.mean) / self.std
            bg = torch.tensor([bg])
            ymin = torch.tensor([ymin])
            ymax = torch.tensor([ymax])
            xmin = torch.tensor([xmin])
            xmax = torch.tensor([xmax])

        return data_input, session_type, bg, forehead, base_forehead, ymin, ymax, xmin, xmax
