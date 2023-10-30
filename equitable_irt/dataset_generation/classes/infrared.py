import os

import cv2
import numpy as np

from ...utils import conv_temp
from ...utils import load_im
from ...utils import raw2temp
from ..face_utils import coords_ir_to_rgb
from ..face_utils import face_bbox
from ..face_utils import ir2rgb
from ..face_utils import load_json


class Infrared:

    WH = (160, 120)
    BG_ROIS = [((0, 20), (0, 20)), ((WH[0] - 20, WH[0]), (0, 20))]
    LOAD_ROIS = [
        'forehead',
        'left_cheek', 'right_cheek',
        'left_canthus', 'right_canthus',
        'left_nose', 'right_nose',
        'left_mouth', 'right_mouth', 'chin',
    ]

    def __init__(self, dataset_dir, name, session, duration, temp_env, units='F'):
        self.dataset_dir = dataset_dir
        self.temp_env = temp_env
        self.units = units
        self.name = name
        self.session = session
        self.duration = duration
        self.session_dir = os.path.join(dataset_dir, 'data', name, session)

        json_path = os.path.join(dataset_dir, 'landmarks', name, f'{session}.json')
        self.landmarks, self.roi_pts, self.invalid = load_json(json_path, duration)
        self.data = self.load()
        self.rois = self.extract_rois()

        # Aligning IR and RGB only depends on camera homography
        self.data, self.landmarks, self.roi_pts = self.align_to_rgb()

        # Extract background
        self.bg = self.extract_background()
        self.nuc_flag = self.find_nuc_regions()
        self.invalid = self.invalid.astype(bool) | self.nuc_flag.astype(bool)
        self.invalid = self.invalid

    def _temp_roi(self, im, bbox, agg_fn=np.mean):
        ylim, xlim = bbox
        region = im[xlim[0]:xlim[1], ylim[0]:ylim[1]]
        return agg_fn(region)

    def load(self):
        n = self.duration
        w, h = Infrared.WH
        ir_images = np.zeros((h, w, n))
        for i in range(n):
            ir_path = os.path.join(self.session_dir, f'ir{i}.png')
            ir = load_im(ir_path, transform=lambda a: raw2temp(a, self.units))
            ir_images[..., i] = ir
        return ir_images

    def extract_rois(self):
        temps = np.zeros((len(Infrared.LOAD_ROIS), self.duration))
        for i in range(self.duration):
            frame = self.data[..., i]
            rois = self.roi_pts[i]
            if len(rois) == 0:
                temps[..., i] = np.nan
            else:
                temps[..., i] = [self._temp_roi(frame, rois[loc]) for loc in Infrared.LOAD_ROIS]
        roi_dict = {}
        for j, loc in enumerate(Infrared.LOAD_ROIS):
            roi_dict[loc] = temps[j]
        return roi_dict

    def align_to_rgb(self):
        (w, h), n = Infrared.WH, self.duration
        data_warp = np.zeros((h, w, n))
        rgb_landmarks = []
        rgb_roi_pts = []
        for i in range(self.duration):
            data_warp[..., i] = cv2.warpPerspective(self.data[..., i], ir2rgb, (w, h))
            if self.landmarks[i] is None or len(self.landmarks[i]) == 0:
                rgb_landmarks += [None]
                rgb_roi_pts += [{}]
                continue
            rgb_landmarks += [coords_ir_to_rgb(self.landmarks[i]).T]
            new_pts = {}
            for k in self.roi_pts[i]:
                pts = self.roi_pts[i][k]
                pts = coords_ir_to_rgb(np.array(pts).T)
                new_pts[k] = np.array([pts[1], pts[0]])
            rgb_roi_pts += [new_pts]
        return data_warp, rgb_landmarks, rgb_roi_pts

    def extract_background(self):
        temp_bgs = np.zeros(self.duration)
        thres = conv_temp(70, 'F', self.units)
        for i in range(self.duration):
            frame = self.data[..., i]
            lms = self.landmarks[i]
            roi = self.roi_pts[i]
            if lms is None or len(lms) == 0 or len(roi) == 0:
                temp_bgs[i] = np.nan
                continue
            (ymin, ymax), (xmin, xmax) = face_bbox(lms, roi)
            ymin = ymin + 10
            left = frame[:ymin, :xmin]
            right = frame[:ymin, xmax:160]

            # IR images are aligned, so patches may contain filler 0 values
            bg = left[left > thres].mean()
            bg += right[right > thres].mean()
            bg /= 2
            temp_bgs[i] = bg
        return temp_bgs

    def find_nuc_regions(self):
        bg = self.bg
        kernel = np.ones(4 * 8, np.uint8)
        thres = np.percentile(bg, 90)
        nuc_flag = (bg > thres).astype(np.uint8)
        nuc_flag = cv2.dilate(nuc_flag, kernel, iterations=1)
        return nuc_flag.reshape(-1)
