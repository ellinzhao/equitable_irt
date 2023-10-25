import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from ..utils import load_im
from ..utils import raw2temp
from ..dataset_generation import Infrared


LOAD_ROIS = Infrared.LOAD_ROIS


class SolarDataset(Dataset):

    def __init__(self, dataset_dir, ids, transform=None, train=False):
        self.dataset_dir = dataset_dir
        self.ids = ids
        self.transform = transform
        self.train = train
        self.loader = lambda path: load_im(path, raw2temp).astype(np.float32)

        labels_data = [self._load_sub_labels(sid) for sid in ids]
        labels, ref_rois = list(zip(*labels_data))
        self.labels = pd.concat(labels)
        self.ref_rois = {self.ids[i]: ref_rois[i] for i in range(len(self.ids))}

    def _load_sub_labels(self, sid):
        fix_dir = lambda x: os.path.join(self.dataset_dir, sid, x)
        base_df = pd.read_csv(fix_dir('base.csv')).iloc[:4 * 30]
        cool_df = pd.read_csv(fix_dir('cool.csv')).iloc[:4 * 30]
        ref_roi_df = pd.read_csv(fix_dir('base_rois.csv'))
        ref_roi = np.array(ref_roi_df.iloc[0])[1:]

        # Combine base and cool labels
        df = pd.concat([base_df, cool_df])
        df['ir_fname'] = df['ir_fname'].apply(fix_dir)
        df['rgb_fname'] = df['rgb_fname'].apply(fix_dir)
        return df, ref_roi

    def load_bbox(self, row):
        pts = []
        for k in LOAD_ROIS:
            ymin, ymax, xmin, xmax = row[[f'{k}_ymin', f'{k}_ymax', f'{k}_xmin', f'{k}_xmax']]
            pts += [(ymin, ymax, xmin, xmax)]
        return np.array(pts)

    def bbox_mask(self, bboxes, shape):
        f = len(bboxes)
        mask = np.zeros((shape[0], shape[1], f))
        for i in range(f):
            ymin, ymax, xmin, xmax = bboxes[i]
            mask[ymin:ymax, xmin:xmax, i] = 1
        return mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.labels.iloc[idx]
        bbox = self.load_bbox(row)
        is_base = row['session_type'] == 'base'

        # Load IR image
        fname = row['ir_fname']
        ir = self.loader(fname)
        ref_rois = self.ref_rois[os.path.basename(os.path.dirname(fname))]
        mask = self.bbox_mask(bbox, ir.shape)

        if self.transform:
            ir = self.transform(ir)
            mask = ToTensor()(mask)
        return ir, ref_rois, mask, is_base
