import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils import load_im
from ..utils import raw2temp


class SolarDataset(Dataset):

    def __init__(self, dataset_dir, ids, transform=None, train=False):
        self.dataset_dir = dataset_dir
        self.ids = ids
        self.transform = transform
        self.train = train
        self.loader = lambda path: load_im(path, raw2temp).astype(np.float32)

        dfs_and_ims = [self._load_sub_labels(sid) for sid in ids]
        labels, ref_ir = list(zip(*dfs_and_ims))
        self.labels = pd.concat(labels)
        self.ref_irs = {self.ids[i]: ref_ir[i] for i in range(len(self.ids))}

    def _load_sub_labels(self, sid):
        sid_dir = os.path.join(self.dataset_dir, sid)
        base_df = pd.read_csv(os.path.join(sid_dir, 'base.csv'))
        cool_df = pd.read_csv(os.path.join(sid_dir, 'cool.csv'))

        # Combine base and cool labels
        df = pd.concat([base_df, cool_df])
        fix_dir = lambda x: os.path.join(sid_dir, os.path.basename(x))
        df['ir_fname'] = df['ir_fname'].apply(fix_dir)
        df['rgb_fname'] = df['rgb_fname'].apply(fix_dir)
        df['sn_fname'] = df['sn_fname'].apply(fix_dir)

        ref_fname = base_df.iloc[4]['ir_fname']
        ref_fname = os.path.basename(ref_fname)
        ref_fname = os.path.join(sid_dir, ref_fname)
        ref_ir = self.loader(ref_fname)
        return df, ref_ir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load IR image
        fname = self.labels['ir_fname'].iloc[idx]
        ir = self.loader(fname)
        # Load avg baseline IR image
        base_ir = self.ref_irs[os.path.dirname(fname)]
        if 'base_' in fname:
            base_ir = ir

        sn_fname = self.labels['sn_fname'].iloc[idx]
        sn = np.load(sn_fname)
        lvec_fname = os.path.join(os.path.dirname(sn_fname), 'lvec.npy')
        lvec = np.load(lvec_fname)
        lvec_full = np.ones((64, 64, 3))
        for i in range(3):
            lvec_full[..., i] = lvec[i]

        # Load RGB image as grayscale
        # rgb_fname = self.labels['rgb_fname'].iloc[idx]
        # rgb = load_im(rgb_fname).astype(np.float32)
        # gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        # Use gray image as placeholder in base_ir to make tforms easier
        # ir = np.dstack([ir, gray])
        # base_ir = np.dstack([base_ir, gray])

        if self.transform:
            ir = self.transform(ir)
            base_ir = self.transform(base_ir)

        sn_tform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])
        if 'base_' in fname:
            label = torch.tensor([1, 0])
        else:
            label = torch.tensor([0, 1])

        return ir, base_ir, sn_tform(sn), sn_tform(lvec_full), label
