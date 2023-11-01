import os

import numpy as np
import cv2
import pandas as pd

from ...utils import temp2raw
from ..face_utils import crop_resize
from ..face_utils import face_bbox
from ..face_utils import frontal_pose
from ..face_utils import surface_normals
from .infrared import Infrared
from .rgb import RGB


class Session:

    DURATION = {
        'cool': 2 * 60 * 4,
        'base': 1 * 60 * 4,
    }

    def __init__(self, dataset_dir, name, session_type, temp_env, save_sn=False, units='F'):
        '''
        Loads both IR and RGB data for the specified session.
        '''
        assert session_type in Session.DURATION
        self.session_type = session_type
        self.duration = Session.DURATION.get(session_type)
        self.units = units
        self.dataset_dir = dataset_dir
        self.name = name
        self.save_sn = save_sn

        self.ir = Infrared(dataset_dir, name, session_type, self.duration, temp_env, units)
        invalid = self.ir.invalid
        landmarks = self.ir.landmarks  # IR is aligned to RGB so landmarks are now in RGB coords
        roi_pts = self.ir.roi_pts  # Same for the ROI points
        self.rgb = RGB(dataset_dir, name, session_type, self.duration, landmarks, roi_pts, invalid)

        # Crop all images to the face bbox
        self.crop_face(self.save_sn)
        self.crop_lms(landmarks, invalid)
        self.invalid = invalid
        self.landmarks = landmarks

    def crop_lms(self, landmarks, invalid):
        resized = []
        for i in range(self.duration):
            lms = landmarks[i]
            if invalid[i] or not frontal_pose(lms):
                resized += [None]
                invalid[i] = True
                continue
            lms = lms[:27].T.copy()
            offset = lms.min(axis=1).astype(int)[:, None]
            lms -= offset
            shape = lms.max(axis=1)[:, None]
            lms = lms / shape * 64
            resized += [lms]
        self.lms_crop = resized

    def crop_face(self, save_sn):
        n = self.duration
        k = 64
        rgb_crop = np.zeros((k, k, 3, n))
        ir_crop = np.zeros((k, k, n))

        for i in range(self.duration):
            if self.rgb.invalid[i] or self.rgb.landmarks[i] is None:
                continue
            rgb_i = self.rgb.data[..., i]
            ir_i = self.ir.data[..., i]
            roi_i = face_bbox(self.rgb.landmarks[i], self.rgb.roi_pts[i])
            rgb_crop[..., i] = crop_resize(rgb_i, roi_i, k)
            ir_crop[..., i] = crop_resize(ir_i, roi_i, k)
        self.ir.crop = ir_crop
        self.rgb.crop = rgb_crop
        if save_sn:
            self.normals = self.get_sn()

    def get_sn(self):
        batch_size = 32
        ims = self.rgb.crop
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

    def generate_dataset(self, idxs):
        save_dir = os.path.join(self.dataset_dir, 'ml_data', self.name)

        roi_df = pd.DataFrame({
            'bg': self.ir.bg,
            'forehead': self.ir.rois['forehead'],
            'invalid': self.ir.invalid,
        })
        roi_df.to_csv(os.path.join(save_dir, f'{self.session_type}_temps.csv'))

        for i in idxs:
            if self.invalid[i]:
                continue
            ir = self.ir.crop[..., i]
            rgb = self.rgb.crop[..., i].astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            raw = temp2raw(ir).astype(np.uint16)

            ir_save_path = f'{self.session_type}_ir{i}.png'
            rgb_save_path = f'{self.session_type}_rgb{i}.jpg'
            cv2.imwrite(os.path.join(save_dir, ir_save_path), raw)
            cv2.imwrite(os.path.join(save_dir, rgb_save_path), rgb)
