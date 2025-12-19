import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.constants import convert_temperature as conv_temp
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from skimage import exposure


TEMP_VMIN_F = 80
TEMP_VMAX_F = 110
TEMP_VMIN_C = conv_temp(TEMP_VMIN_F, 'F', 'C')
TEMP_VMAX_C = conv_temp(TEMP_VMAX_F, 'F', 'C')

MEAN_C = [32.5896, 32.5896]
STD_C = [3.3411, 3.3411]

INV_FN = lambda x: x


SCALE_FN = lambda x: (x - MEAN_C[0]) / STD_C[0]
UNSCALE_FN = lambda x: x * STD_C[0] + MEAN_C[0]
BG_THRES_C = SCALE_FN(28)  # 82.4 F

CROP_FN = lambda x: crop(x, 0, 8, 54, 48)

TRAIN_TFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(MEAN_C, STD_C),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Lambda(CROP_FN),
    transforms.Resize((32, 32)),
])

VAL_TFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(MEAN_C, STD_C),
    transforms.Lambda(CROP_FN),
    transforms.Resize((32, 32)),
])


def raw2temp(raw, units='F'):
    return conv_temp((raw - 27315) / 100, 'C', units)


def load_im(path, transform=None):
    if not transform:
        transform = lambda raw: raw
    return transform(cv2.imread(path, -1))


def simulate_fever(ir, base_ir, ir_mask, base_mask, bg, base_bg):
    blood_scaling = random.uniform(1, 2)
    offset = random.uniform(1, 3)

    ir_norm = exposure.rescale_intensity(ir, 'image')
    ir_gamma = exposure.adjust_gamma(ir_norm, 1.75)

    ir = ir + ir_gamma * blood_scaling + offset
    ir[ir_mask] = bg  # TEMP_VMIN_F
    base_ir = base_ir + ir_gamma * blood_scaling + offset
    base_ir[base_mask] = base_bg  # TEMP_VMIN_F

    fever = offset  # (ir - ir_original).sum() / ir_mask.sum()
    return ir, base_ir, fever


class SolarDataset(Dataset):

    def __init__(self, dataset_dir, dataset_dict, ids, num_load=4 * 60 * 5, transform=None,
                 train=False, fever_prob=0, session_filter=None, clamp_sl=True):
        self.dataset_dir = dataset_dir
        self.dataset_dict = dataset_dict
        self.ids = ids
        self.transform = transform
        self.train = train
        self.num_load = num_load
        self.fever_prob = fever_prob
        self.session_filter = session_filter
        self.clamp_sl = clamp_sl

        # Mean and std for IR data
        self.mean = MEAN_C[0]
        self.std = STD_C[0]
        self.loader = lambda path: self.dataset_dict[path].copy()
        self.labels = pd.concat([self._load_sub_labels(sid) for sid in ids])

    def _fname_index(self, x):
        session, i = re.findall(r'([a-z]+)_ir(\d+).png', os.path.basename(x))[0]
        return f'{session}_{i}'

    def _load_sub_labels(self, sid):
        sid_dir = lambda x: os.path.join(self.dataset_dir, sid, x)

        # The labels csv has the paired file names
        df = pd.read_csv(sid_dir('full_labels.csv')).iloc[:self.num_load]
        for c in ['ir_fname', 'base_ir_fname']:
            df[c] = df[c].apply(sid_dir)
        df['session_i'] = df['ir_fname'].apply(self._fname_index)
        df['session_i'] = sid + '_' + df['session_i']
        df.set_index('session_i', inplace=True)

        mask = df.index.str.contains('_base_')
        base_forehead_mean = df[mask]['base_forehead'].mean()
        df['base_forehead_mean'] = base_forehead_mean

        if self.train:
            mask = df.index.str.contains('_base_')
            df = pd.concat([df, df[mask]])

        if self.session_filter:
            mask = df.index.str.contains(f'_{self.session_filter}_')
            df = df[mask]

        return df

    def __len__(self):
        return len(self.labels)

    def get_fname(self, idx):
        row = self.labels.iloc[idx]
        return row.index

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.labels.iloc[idx]

        bg = row['cool_bg']
        base_bg = row['base_bg']
        forehead = row['cool_forehead']
        base_forehead = row['base_forehead_mean']
        oral = row['oral']

        # Load IR image
        ir_fname = row['ir_fname']
        base_ir_fname = row['base_ir_fname']
        ir = self.loader(ir_fname)
        base_ir = self.loader(base_ir_fname)
        is_base = base_ir_fname == ir_fname
        sl = 0 if is_base else forehead - base_forehead

        # Clip background to a uniform value (VMIN)
        ir_mask = ir <= bg
        ir[ir_mask] = bg  # TEMP_VMIN_F
        base_mask = base_ir <= base_bg
        base_ir[base_mask] = base_bg  # TEMP_VMIN_F

        # If SL too low, consider the image to be a baseline image
        if self.clamp_sl and sl <= 0.1:
            ir = base_ir
            ir_mask = base_mask
            is_base = 1
            sl = 0
        session_type = [is_base, not is_base]

        # Add fever offset to both images
        if is_base and random.uniform(0, 1) < self.fever_prob:
            ir, base_ir, fever_offset = simulate_fever(ir, base_ir, ir_mask, base_mask, bg, base_bg)
            oral = oral + fever_offset

        # Convert to SI units
        im = np.dstack([ir, base_ir])
        im = np.clip(im, TEMP_VMIN_F, TEMP_VMAX_F)
        im = conv_temp(im, 'F', 'C')
        sl = conv_temp(sl, 'F', 'C') - conv_temp(0, 'F', 'C')
        oral = conv_temp(oral, 'F', 'C')

        if self.transform:
            im = self.transform(im)
            session_type = torch.tensor(session_type, dtype=int)
            sl = torch.tensor(sl)
        return im, session_type, idx
