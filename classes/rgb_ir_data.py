import json
import os

import cv2
import numpy as np

from utils import load_im
from utils import raw2temp


class RGB_IR_Data:

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

    LANDMARK_LOCS = [
        'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip',
        'left_eye', 'right_eye', 'top_lip', 'bottom_lip',
    ]

    FACE_ROI = ((51, 103), (27, 99))  # Derived from the ref.json. If ref.json changes, update this.

    def __init__(self, dataset_dir, name, session, temp_env, rh):
        assert session in RGB_IR_Data.SESSION_DURATION.keys()
        self.session = session
        self.duration = RGB_IR_Data.SESSION_DURATION.get(session)
        self.dataset_dir = dataset_dir
        self.name = name
        self.temp_env = temp_env
        self.rh = rh
        self._load_landmarks()
        self._load_ref_landmarks()
        self._load_ir()

    def _load_ir(self, correct_nuc=True, normalize_pos=True, crop=True):
        n = self.duration
        if crop:
            ylim, xlim = RGB_IR_Data.FACE_ROI
            shape = (xlim[1] - xlim[0], ylim[1] - ylim[0], n)
        else:
            w, h = RGB_IR_Data.WH
            shape = (h, w, n)
        images = np.zeros(shape)
        for i in range(n):
            im_path = os.path.join(self.dataset_dir, 'data', self.name, self.session, f'ir{i}.png')
            im = load_im(im_path, transform=raw2temp)
            if correct_nuc:
                im = self.correct_nuc(im)
            if normalize_pos:
                im = self.normalize_face_position(im, self.landmarks[i], crop=crop)
            images[..., i] = im
        self.ir_data = images

    def _load_landmarks(self):
        # List: len=n_ims, each item is array shape (n, 2)
        json_path = os.path.join(self.dataset_dir, 'landmarks', self.name, f'{self.session}.json')
        landmark_arr = []
        with open(json_path) as json_file:
            json_data = json.load(json_file)

        for i in range(self.duration):
            json_i = json_data.get(f'rgb{i}.jpg', None)
            if json_i is None:
                print('cant find json')
                json_i = landmark_arr[-1]
            landmark_arr += [self._landmark_json_to_arr(json_i['landmarks'])]
        self.landmarks = landmark_arr

    def _load_ref_landmarks(self):
        json_path = os.path.join(self.dataset_dir, 'landmarks', 'ref.json')
        with open(json_path) as json_file:
            json_data = json.load(json_file)
        self.ref_landmarks = self._landmark_json_to_arr(json_data)

    def _landmark_json_to_arr(self, dct):
        all_points = []
        for loc in self.LANDMARK_LOCS:
            points = dct.get(loc)
            points = np.array(points).T  # output: (n_points, 2)
            all_points.extend(points)
        return np.array(all_points)

    def region_temp(self, im, location, agg_fn):
        pass

    def bbox_temp(self, im, bbox, agg_fn=np.mean):
        ylim, xlim = bbox
        return agg_fn(im[xlim[0]:xlim[1], ylim[0]:ylim[1]])

    def correct_nuc(self, im):
        # Calibrate temperature based on background
        # NUC correction must occur before any image transforms
        bg = [self.bbox_temp(im, bbox) for bbox in self.BG_ROIS]
        bg = np.mean(bg)
        return im - (bg - self.temp_env)

    def normalize_face_position(self, im, src_pts, crop):
        dst_pts = self.ref_landmarks
        h, _ = cv2.findHomography(src_pts, dst_pts)
        im_warp = cv2.warpPerspective(im, h, (im.shape[1], im.shape[0]))
        if crop:
            ylim, xlim = RGB_IR_Data.FACE_ROI
            im_warp = im_warp[xlim[0]:xlim[1], ylim[0]:ylim[1]]
        return im_warp
