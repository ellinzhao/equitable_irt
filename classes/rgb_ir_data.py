import os

import numpy as np

from utils import load_im
from utils import raw2temp


class RGB_IR_Data:
    # normalize all face positions to some known config?

    SESSION_DURATION = {
        'cool': 5 * 60 * 4,
        'base': 1 * 60 * 4,
    }

    WH = (160, 120)

    BG_ROIS = [
        ((0, 20), (0, 20)),  # top left
        ((WH[0] - 20, WH[0]), (0, 20)),  # top right
        # ((0, WH[0]), (0, WH[1])),  # whole image?
    ]

    ROI_KEYS = [
        'forehead', 'chin',
        'left_cheek', 'right_cheek',
        'left_canthus', 'right_canthus',
        'left_eye_corner', 'right_eye_corner',
        'left_nose', 'right_nose',
        'left_mouth', 'right_mouth',
    ]

    def __init__(self, data_path, temp_env, rh):
        self.data_path = data_path
        self.temp_env = temp_env
        self.rh = rh
        self.session_data = {}
        self._load_ir('base')
        self._load_ir('cool')

    def _load_ir(self, session='cool'):
        print(f'Loading {session} data...')
        n = self.SESSION_DURATION[session]
        images = np.zeros((120, 160, n))
        for i in range(n):
            im_path = os.path.join(self.data_path, session, f'ir{i}.png')
            im = load_im(im_path, transform=raw2temp)
            im = self.nuc_correction(im)
            images[..., i] = im
        self.session_data[session] = images

    def region_temp(self, im, bbox, agg_fn=np.mean):
        xlim, ylim = bbox
        return agg_fn(im[ylim[0]:ylim[1], xlim[0]:xlim[1]])

    def nuc_correction(self, im):
        # Calibrate temperature based on background
        bg = [self.region_temp(im, bbox) for bbox in self.BG_ROIS]
        bg = np.mean(bg)
        return im - (bg - self.temp_env)
