import os

import numpy as np
import cv2
import pandas as pd

from ...utils import temp2raw
from ..face_utils import crop_resize
from ..face_utils import face_bbox
from ..face_utils import flatten_rois
from ..face_utils import surface_normals
from .infrared import Infrared
from .rgb import RGB


class Session:

    DURATION = {
        'cool': 1 * 60 * 4,
        'base': 1 * 60 * 4,
    }

    def __init__(self, dataset_dir, name, session_type, temp_env, units='F'):
        '''
        Loads both IR and RGB data for the specified session.
        '''
        assert session_type in Session.DURATION
        self.session_type = session_type
        self.duration = Session.DURATION.get(session_type)
        self.units = units
        self.dataset_dir = dataset_dir
        self.name = name

        self.ir = Infrared(dataset_dir, name, session_type, self.duration, temp_env, units)
        self.invalid = self.ir.invalid
        landmarks = self.ir.landmarks  # IR is aligned to RGB so landmarks are now in RGB coords
        roi_pts = self.ir.roi_pts  # Same for the ROI points
        self.rgb = RGB(dataset_dir, name, session_type, self.duration,
                       landmarks, roi_pts, self.invalid)

    def crop_face(self, save_sn):
        n = self.duration
        k = 64
        rgb_warped = np.zeros((k, k, 3, n))
        ir_warped = np.zeros((k, k, n))
        crop_offset = np.zeros((4, n))

        for i in range(self.duration):
            if self.rgb.invalid[i] or self.rgb.landmarks[i] is None:
                continue
            rgb_i = self.rgb.data[..., i]
            ir_i = self.ir.data[..., i]
            roi_i = face_bbox(self.rgb.landmarks[i], self.rgb.roi_pts[i])
            crop_offset[..., i] = np.array(roi_i).flatten()
            rgb_warped[..., i] = crop_resize(rgb_i, roi_i, k)
            ir_warped[..., i] = crop_resize(ir_i, roi_i, k)
        self.ir.warped = ir_warped
        self.rgb.warped = rgb_warped
        self.crop_offset = crop_offset
        if save_sn:
            self.normals = self.get_sn()

    def get_sn(self):
        batch_size = 32
        ims = self.rgb.warped
        sns = []
        for i in range(0, self.duration, batch_size):
            batch = ims[..., i:i + batch_size]
            batch = batch.transpose(3, 0, 1, 2)  # b, w, h, c
            batch_sns = surface_normals(batch)
            sns += list(batch_sns)
        return sns

    def save_roi_values(self):
        mean_temps = []
        invalid = self.ir.invalid
        for k in self.ir.LOAD_ROIS:
            temp = np.nanmean(self.ir.rois[k][~invalid])
            mean_temps += [temp]

        mean_temps = np.array(mean_temps).reshape(1, -1)
        df = pd.DataFrame(data=mean_temps, columns=self.ir.LOAD_ROIS)
        save_dir = os.path.join(self.dataset_dir, 'ml_data', self.name,
                                f'{self.session_type}_rois.csv')
        df.to_csv(save_dir)
        return 1

    def generate_dataset(self, save_sn):
        session_labels = []
        n = self.duration
        save_dir = os.path.join(self.dataset_dir, 'ml_data', self.name)
        for i in range(n):
            if self.invalid[i]:
                continue

            labels = []
            roi = self.ir.roi_pts[i]
            ir = self.ir.warped[..., i]
            rgb = self.rgb.warped[..., i].astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            raw = temp2raw(ir).astype(np.uint16)

            ir_save_path = f'{self.session_type}_ir{i}.png'
            rgb_save_path = f'{self.session_type}_rgb{i}.jpg'
            cv2.imwrite(os.path.join(save_dir, ir_save_path), raw)
            cv2.imwrite(os.path.join(save_dir, rgb_save_path), rgb)
            labels = [ir_save_path, rgb_save_path]

            if save_sn:
                sn_save_path = f'{self.session_type}_sn{i}.npy'
                normals = self.normals[i]
                normals = cv2.resize(normals, (64, 64))
                np.save(os.path.join(save_dir, sn_save_path), normals)
                labels += [sn_save_path]

            labels += [self.session_type]
            labels += flatten_rois(roi, self.crop_offset[..., i], self.ir.LOAD_ROIS)
            session_labels += [labels]

        session_labels = np.array(session_labels)
        cols = self._dataset_columns(save_sn)
        df = pd.DataFrame(data=session_labels, columns=cols)

        # The last columns are the bbox points and should be integer values.
        k = 4 * len(self.ir.LOAD_ROIS)
        df.iloc[:, -k:] = df.iloc[:, -k:].astype(int)
        return df

    def _dataset_columns(self, save_sn):
        '''
        rgb_fname, ir_fname, [sn_fname], session_type, forehead_ymin, forehead_ymax, ...
        '''
        cols = ['ir_fname', 'rgb_fname']
        if save_sn:
            cols += ['sn_fname']
        cols += ['session_type']
        for k in self.ir.LOAD_ROIS:
            cols += [f'{k}_ymin', f'{k}_ymax', f'{k}_xmin', f'{k}_xmax']
        return cols
