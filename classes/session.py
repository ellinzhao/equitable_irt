import os

import numpy as np
import cv2

from .infrared import Infrared
from .rgb import RGB
from face_utils import align_rgb
from face_utils import crop_resize
from utils import temp2raw


class Session:

    DURATION = {
        'cool': 1 * 60 * 4,
        'base': 1 * 60 * 4,
    }

    def __init__(self, dataset_dir, name, session_type, temp_env, units='F'):
        assert session_type in Session.DURATION
        self.session_type = session_type
        self.duration = Session.DURATION.get(session_type)
        self.units = units
        self.dataset_dir = dataset_dir
        self.name = name

        self.ir = Infrared(dataset_dir, name, session_type, self.duration, temp_env, units)
        self.invalid = self.ir.invalid
        rgb_landmarks = self.ir.landmarks  # IR is aligned to RGB so landmarks are now in RGB coords
        self.rgb = RGB(dataset_dir, name, session_type, self.duration, rgb_landmarks, self.invalid)

    def align_to_ref(self, rgb_ref, roi):
        n = self.duration
        k = 64
        rgb_warped = np.zeros((k, k, 3, n))
        ir_warped = np.zeros((k, k, n))

        for i in range(self.duration):
            if self.rgb.invalid[i]:
                continue
            rgb_i = self.rgb.data[..., i]
            ir_i = self.ir.data[..., i]
            h = align_rgb(rgb_i.astype(np.uint8), rgb_ref.astype(np.uint8))
            if h is None:
                continue
            rgb_warp = cv2.warpPerspective(rgb_i, h, (rgb_i.shape[1], rgb_i.shape[0]))
            ir_warp = cv2.warpPerspective(ir_i, h, (ir_i.shape[1], ir_i.shape[0]))
            rgb_warped[..., i] = crop_resize(rgb_warp, roi, k)
            ir_warped[..., i] = crop_resize(ir_warp, roi, k)
        self.ir.warped = ir_warped
        self.rgb.warped = rgb_warped

    def generate_dataset(self):
        fnames = []
        n = int(self.duration / 1)
        save_dir = os.path.join(self.dataset_dir, 'ml_data', self.name)
        for i in range(n):
            start, end = i * 1, (i + 1) * 1
            idxs = np.arange(start, end, dtype=int)
            skip = np.any(self.invalid[idxs])
            if skip:
                continue
            ir = np.mean(self.ir.warped[..., idxs], axis=-1)
            rgb = np.mean(self.rgb.warped[..., idxs], axis=-1).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            raw = temp2raw(ir).astype(np.uint16)

            ir_save_path = os.path.join(save_dir, f'{self.session_type}_ir{i}.png')
            cv2.imwrite(ir_save_path, raw)
            rgb_save_path = os.path.join(save_dir, f'{self.session_type}_rgb{i}.png')
            cv2.imwrite(rgb_save_path, rgb)

            fnames += [(ir_save_path, rgb_save_path)]
        return np.array(fnames)
