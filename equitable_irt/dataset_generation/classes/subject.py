import os

import pandas as pd

from ...utils import conv_temp
from .session import Session


class Subject:

    def __init__(self, dataset_dir, name, units='F', save_sn=False):
        self.dataset_dir = dataset_dir
        self.name = name
        self.units = units
        self.save_sn = save_sn
        self.load_csv()
        self.base = Session(dataset_dir, name, 'base', self.temp_env, units=units)
        self.cool = Session(dataset_dir, name, 'cool', self.temp_env, units=units)
        self.base.crop_face(save_sn)
        self.cool.crop_face(save_sn)

    def load_csv(self):
        csv_path = os.path.join(self.dataset_dir, 'data.csv')
        df = self.clean_df(csv_path)
        data_dir = self.name
        row = df[df['dir'] == data_dir]

        # Skin tone
        self.colorimeter = row[['E', 'M']]
        self.fst = row['FST']

        # Environment
        temp_env = [72.5] if row['temp_env'].empty else row['temp_env']
        self.temp_env = conv_temp(temp_env, 'F', self.units)[0]
        self.temp_env = min(self.temp_env, conv_temp(72.5, 'F', self.units))
        self.rh = row['rh']

        # Body temperature (GT and IRTs)
        self.oral = conv_temp(row['Oral 0'], 'F', self.units)
        self.irt = {}
        for device in ['Purple', 'Sejoy', 'ADC']:
            cols = [f'{device} {i}' for i in range(3)]
            self.irt[device] = conv_temp(row[cols], 'F', self.units)
        return 1

    def clean_df(self, path):
        cols = [
            'dir', 'FST', 'E', 'M', 'temp_env', 'rh',
            'Oral 0', 'Purple 0', 'ADC 0', 'Sejoy 0',
            'Purple 1', 'ADC 1', 'Sejoy 1',
            'Purple 2', 'ADC 2', 'Sejoy 2',
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

        # The Session objects generate dataframes and save all the images
        base_df = self.base.generate_dataset(save_sn)
        cool_df = self.cool.generate_dataset(save_sn)

        save_path = os.path.join(save_dir, 'base.csv')
        base_df.to_csv(save_path, index=False)

        save_path = os.path.join(save_dir, 'cool.csv')
        cool_df.to_csv(save_path, index=False)
