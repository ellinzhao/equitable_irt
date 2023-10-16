import os

import cv2
import numpy as np
import pandas as pd
from scipy.constants import convert_temperature as conv_temp

from .session import Session
from face_utils import crop_resize
from face_utils import surface_normals
from utils import temp2raw


class Subject:

    def __init__(self, dataset_dir, name, units='F'):
        self.dataset_dir = dataset_dir
        self.name = name
        self.units = units
        self.load_csv()
        self.base = Session(dataset_dir, name, 'base', self.temp_env, units=units)
        self.cool = Session(dataset_dir, name, 'cool', self.temp_env, units=units)

        # Align both sessions to one base image
        ref_i, ref_rgb, face_roi = self.base.rgb.random_im()
        self.base.align_to_ref(ref_rgb, face_roi)
        self.cool.align_to_ref(ref_rgb, face_roi)

        # Surface normal model take images of size 256x256
        ref_rgb_sn = crop_resize(ref_rgb, face_roi, 256)
        self.ref_sn = cv2.resize(surface_normals(ref_rgb_sn), (64, 64))
        self.ref_rgb = crop_resize(ref_rgb, face_roi, 64)
        ref_ir = self.base.ir.warped[..., ref_i:ref_i + 1]
        self.ref_ir = np.mean(ref_ir, axis=-1)

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
        return df

    def save_dataset(self):
        save_dir = os.path.join(self.dataset_dir, 'ml_data', self.name)
        os.mkdir(save_dir)

        ref_ir = temp2raw(self.ref_ir).astype(np.uint16)
        ref_rgb = self.ref_rgb.astype(np.uint8)
        ref_rgb = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, 'ref.png'), ref_ir)
        cv2.imwrite(os.path.join(save_dir, 'ref.jpg'), ref_rgb)
        np.save(os.path.join(save_dir, 'sn.npy'), self.ref_sn)

        base_fnames = self.base.generate_dataset()
        cool_fnames = self.cool.generate_dataset()

        save_path = os.path.join(save_dir, 'base.csv')
        base_df = pd.DataFrame({
            'ir_fname': base_fnames[:, 0],
            'rgb_fname': base_fnames[:, 1],
        })
        base_df.to_csv(save_path, index=False)

        save_path = os.path.join(save_dir, 'cool.csv')
        cool_df = pd.DataFrame({
            'ir_fname': cool_fnames[:, 0],
            'rgb_fname': cool_fnames[:, 1],
        })
        cool_df.to_csv(save_path, index=False)
