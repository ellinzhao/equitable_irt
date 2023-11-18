import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ...utils import conv_temp
from .session import Session


class Subject:

    def __init__(self, dataset_dir, name, units='F', save_sn=False):
        self.dataset_dir = dataset_dir
        self.name = name
        self.units = units
        self.save_sn = save_sn
        self.df = self.load_csv()
        self.base = Session(dataset_dir, name, 'base', self.temp_env, save_sn, units=units)
        self.cool = Session(dataset_dir, name, 'cool', self.temp_env, save_sn, units=units)

    def match_lms(self):
        framei_1 = np.where(self.base.invalid != 1)[0]
        framei_2 = np.where(self.cool.invalid != 1)[0]
        lm1 = [self.base.lms_crop[i] for i in framei_1]
        lm2 = [self.cool.lms_crop[i] for i in framei_2]
        lm1 = np.array(lm1).reshape(len(lm1), -1)
        lm2 = np.array(lm2).reshape(len(lm2), -1)

        dists = cdist(lm1, lm2, 'euclidean')
        min_idx = np.argmin(dists, axis=0).astype(int)
        min_dist = dists[min_idx, np.arange(dists.shape[1])]  # indices of best base match
        mask = min_dist < 13

        i1 = framei_1[min_idx[mask]]
        i2 = framei_2[mask]
        print('Num matches: ', len(i1))
        return i1, i2

    def load_csv(self):
        csv_path = os.path.join(self.dataset_dir, 'data.csv')
        df = self.clean_df(csv_path)
        data_dir = self.name
        row = df[df['dir'] == data_dir]

        # Skin tone
        self.colorimeter = row[['E', 'M']]
        self.fst = 3 if row['FST'].empty else row['FST'].item()

        # Environment
        temp_env = [72.5] if row['temp_env'].empty else row['temp_env']
        self.temp_env = conv_temp(temp_env, 'F', self.units)[0]
        self.temp_env = min(self.temp_env, conv_temp(72.5, 'F', self.units))
        self.rh = 40 if row['rh'].empty else row['rh'].item()
        self.sun = row['Sun 1'].item()

        # Body temperature (GT and IRTs)
        self.oral = conv_temp(row['Oral 0'], 'F', self.units).item()
        self.irt = {}
        for device in ['Purple', 'Sejoy', 'ADC']:
            cols = [f'{device} {i}' for i in range(3)]
            vals = [float(row[c].item()) for c in cols]
            self.irt[device] = conv_temp(vals, 'F', self.units).flatten()
        return df

    def clean_df(self, path):
        cols = [
            'dir', 'FST', 'E', 'M', 'temp_env', 'rh',
            'Oral 0', 'Purple 0', 'ADC 0', 'Sejoy 0',
            'Purple 1', 'ADC 1', 'Sejoy 1',
            'Purple 2', 'ADC 2', 'Sejoy 2', 'Sun 1',
        ]
        dateparse = '%m-%d %H:%M'
        df = pd.read_csv(path, header=1, parse_dates=[5], date_format=dateparse)
        df = df.dropna(subset='Colorimeter (E,M)')

        dates = pd.to_datetime(df['Datetime'], format='%m-%d %I:%M').dt
        df['name_lower'] = df['Name'].str.lower()
        df['month'] = dates.month
        df['day'] = dates.day

        df['dir'] = df[['month', 'day', 'name_lower']].apply(
            lambda x: f'{str(x[0]).zfill(2)}{str(x[1]).zfill(2)}_{x[2]}',
            axis=1
        )
        df = df.set_index('Subject ID')

        df[['E', 'M']] = df['Colorimeter (E,M)'].str.split(', ', expand=True)
        df['E'] = pd.to_numeric(df['E'])
        df['M'] = pd.to_numeric(df['M'])

        df[['temp_env', 'rh']] = df['Air temp, RH'].str.split(', ', expand=True)
        df['temp_env'] = pd.to_numeric(df['temp_env'])
        df['rh'] = pd.to_numeric(df['rh'])

        df = df[cols]
        return df

    def save_dataset(self, save_sn=False):
        save_dir = os.path.join(self.dataset_dir, 'ml_data', self.name)
        os.mkdir(save_dir)

        base_idx, cool_idx = self.match_lms()
        base_ir_fname = [f'base_ir{i}.png' for i in base_idx]
        cool_ir_fname = [f'cool_ir{i}.png' for i in cool_idx]
        # Add base images where the ground truth is itself
        base_unique = list(set(base_ir_fname))
        cool_ir_fname += base_unique
        base_ir_fname += base_unique

        # Repeat for RGB file names
        base_rgb_fname = [f'base_rgb{i}.jpg' for i in base_idx]
        cool_rgb_fname = [f'cool_rgb{i}.jpg' for i in cool_idx]
        base_unique = list(set(base_rgb_fname))
        cool_rgb_fname += base_unique
        base_rgb_fname += base_unique

        # Save dataframe with the fnames of paired images
        df = pd.DataFrame({
            'ir_fname': cool_ir_fname, 'rgb_fname': cool_rgb_fname,
            'base_ir_fname': base_ir_fname, 'base_rgb_fname': base_rgb_fname,
        })
        df.to_csv(os.path.join(save_dir, 'label.csv'), index=False)

        # Save subject metadata (skin tone info, environment, etc)
        metadata = [self.temp_env, self.rh, self.fst]
        metadata += [self.colorimeter['E'].item(), self.colorimeter['M'].item()]
        metadata += [self.oral]
        cols = ['temp_env', 'rh', 'fst', 'E', 'M', 'oral']
        for device in ['Purple', 'Sejoy', 'ADC']:
            cols += [f'{device} {i}' for i in range(3)]
            metadata += list(self.irt[device])
        meta_df = pd.DataFrame(np.array(metadata).reshape(1, -1), columns=cols)
        meta_df.to_csv(os.path.join(save_dir, 'metadata.csv'), index=False)

        # The Session objects saves all the images
        self.base.generate_dataset(np.unique(base_idx))
        self.cool.generate_dataset(np.unique(cool_idx))
