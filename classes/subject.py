import os

import numpy as np
import pandas as pd
from scipy.constants import convert_temperature as conv_temp

from .rgb_ir_data import RGB_IR_Data


def _load_df(path):
    cols = [
        'dir', 'FST', 'E', 'M', 'temp_env', 'rh',
        'Oral 0', 'Purple 0', 'ADC 0', 'Sejoy 0',
        'Purple 1', 'ADC 1', 'Sejoy 1',
        'Purple 2', 'ADC 2', 'Sejoy 2',
    ]
    dateparse = '%m-%d %H:%M'
    df = pd.read_csv(path, header=1, parse_dates=[5], date_format=dateparse)
    df = df.dropna(subset='Colorimeter (E,M)')
    df = df[3:]

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
    df = df[~df.index.isin([17, 41])]  # Remove bad data
    df['fps'] = [2 if '1122_' in dir else 4 for dir in df['dir']]
    return df


class Subject:

    def __init__(self, dataset_dir, name, units='F'):
        self.dataset_dir = dataset_dir
        self.name = name
        self.units = units
        self._load_csv_data()
        self.base = RGB_IR_Data(dataset_dir, name, 'base', self.temp_env)
        self.cool = RGB_IR_Data(dataset_dir, name, 'cool', self.temp_env)

    def _load_csv_data(self):
        csv_path = os.path.join(self.dataset_dir, 'data.csv')
        df = _load_df(csv_path)
        data_dir = self.name
        row = df[df['dir'] == data_dir]

        # Skin tone
        self.colorimeter = row[['E', 'M']]
        self.fst = row['FST']

        # Environment
        temp_env = 72.5 if row['temp_env'].empty else row['temp_env']
        self.temp_env = conv_temp(temp_env, 'F', self.units)[0]
        self.temp_env = min(self.temp_env, conv_temp(73, 'F', self.units))
        self.rh = row['rh']

        # Body temperature (GT and IRTs)
        self.oral = conv_temp(row['Oral 0'], 'F', self.units)
        self.irt = {}
        for device in ['Purple', 'Sejoy', 'ADC']:
            cols = [f'{device} {i}' for i in range(3)]
            self.irt[device] = conv_temp(row[cols], 'F', self.units)
        return 1

    def save_dataset(self):
        os.mkdir(os.path.join(self.dataset_dir, 'ml_data', self.name))
        self._save_csv()
        self.base.save_ir()
        self.cool.save_ir()

    def _save_csv(self):
        base_gt = self.base.rois['forehead'].mean()

        save_path = os.path.join(self.dataset_dir, 'ml_data', self.name, 'base.csv')
        base_df = pd.DataFrame({
            'fname': np.array([f'base_ir{i}.png' for i in range(self.base.duration)]),
            'base': np.full(self.base.duration, base_gt),
            'delta': np.zeros(self.base.duration),
            'nuc_flag': self.base.nuc_flag,
        })
        base_df.to_csv(save_path, index=False)

        delta_arr = self.cool.rois['forehead'] - base_gt
        save_path = os.path.join(self.dataset_dir, 'ml_data', self.name, 'cool.csv')
        cool_df = pd.DataFrame({
            'fname': np.array([f'cool_ir{i}.png' for i in range(self.cool.duration)]),
            'base': np.full(self.cool.duration, base_gt),
            'delta': delta_arr,
            'nuc_flag': self.cool.nuc_flag,
        })
        cool_df.to_csv(save_path, index=False)
