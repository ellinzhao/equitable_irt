import json
import os

import cv2
import numpy as np
import scipy.signal as sps

from utils import bgr2rgb
from utils import coords_ir_to_rgb
from utils import ir2rgb
from utils import load_im
from utils import raw2temp
from utils import temp2raw


class RGB_IR_Data:

    SESSION_DURATION = {
        'cool': 5 * 60 * 4,
        'base': 1 * 60 * 4,
    }

    WH = (160, 120)

    CROP_WH = (60, 90)

    BG_ROIS = [
        ((0, 20), (0, 20)),  # top left
        ((WH[0] - 20, WH[0]), (0, 20)),  # top right
        # ((0, WH[0]), (0, WH[1])),  # whole image?
    ]

    LANDMARK_LOCS = [
        'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip',
        'left_eye', 'right_eye', 'top_lip', 'bottom_lip',
    ]

    BBOX_LOCS = [
        'forehead', 'left_cheek', 'right_cheek', 'left_canthus', 'right_canthus',
        'left_eye_corner', 'right_eye_corner', 'left_nose', 'right_nose',
        'left_mouth', 'right_mouth', 'chin',
    ]

    def __init__(self, dataset_dir, name, session, temp_env,
                 correct_nuc=True, align_ir_rgb=True, rois=['forehead']):
        assert session in RGB_IR_Data.SESSION_DURATION.keys()
        self.name = name
        self.session = session
        self.temp_env = temp_env
        self.duration = RGB_IR_Data.SESSION_DURATION.get(session)
        self.dataset_dir = dataset_dir

        self.ir2rgb = ir2rgb
        self.landmarks, self.bboxes = self._load_landmarks()
        self.ref_landmarks = self._load_ref_landmarks()
        self.rois, self.ir, self.rgb = self._load_ir(correct_nuc, align_ir_rgb, rois)
        self.nuc_flag = self._find_nuc()

    def _find_nuc(self):
        n = self.duration
        bg = self.rois['bg']
        peaks, _ = sps.find_peaks(bg, height=self.temp_env + 1.5, width=4 * 5)
        nuc_i = []
        for p in peaks:
            nuc_i.extend(np.arange(max(p - 4 * 5, 0), min(p + 4 * 10, n)))
        nuc_i = np.array(nuc_i, dtype=int)
        nuc_flag = np.zeros(n)
        nuc_flag[nuc_i] = 1
        return nuc_flag

    def _load_ir(self, correct_nuc, align_ir_rgb, rois):
        n = self.duration
        w, h = RGB_IR_Data.CROP_WH
        ir_images = np.zeros((h, w, n))
        rgb_images = np.zeros((h, w, 3, n))
        roi_temps = np.zeros((len(rois), n))
        bg_temps = np.zeros(n)

        for i in range(n):
            session_dir = os.path.join(self.dataset_dir, 'data', self.name, self.session)
            ir_path = os.path.join(session_dir, f'ir{i}.png')
            rgb_path = os.path.join(session_dir, f'rgb{i}.jpg')
            ir = load_im(ir_path, transform=raw2temp)
            rgb = load_im(rgb_path, transform=bgr2rgb)

            # NUC correction and ROI calc must occure before image alignment
            if correct_nuc:
                ir, bg = self.correct_nuc(ir)
                bg_temps[i] = bg
            if rois:
                roi_temps[..., i] = self._load_rois(ir, self.bboxes[i], rois)

            if align_ir_rgb:
                ir, rgb = self.align_ir_to_rgb(ir, rgb, self.landmarks[i])
            else:
                rgb = rgb[:h, :w]
            ir_images[..., i] = ir
            rgb_images[..., i] = rgb

        roi_temps = self._roi_arr_to_dict(roi_temps, rois)
        roi_temps['bg'] = bg_temps
        return roi_temps, ir_images, rgb_images

    def _load_rois(self, im, bboxes, rois):
        return [self.bbox_temp(im, bboxes[loc]) for loc in rois]

    def _roi_arr_to_dict(self, arr, rois):
        roi_dict = {}
        for i, loc in enumerate(rois):
            roi_dict[loc] = arr[i]
        return roi_dict

    def _load_landmarks(self):
        # List: len=n_ims, each item is array shape (n, 2)
        json_path = os.path.join(self.dataset_dir, 'landmarks', self.name, f'{self.session}.json')
        landmark_arr = []
        bbox_arr = []
        with open(json_path) as json_file:
            json_data = json.load(json_file)

        for i in range(self.duration):
            json_i = json_data.get(f'rgb{i}.jpg', None)
            if json_i is None:
                print('cant find json')
                json_i = landmark_arr[-1]
            landmark_arr += [self._landmark_json_to_arr(json_i['landmarks'])]
            bbox_arr += [json_i['rois']]
        return landmark_arr, bbox_arr

    def _load_ref_landmarks(self):
        json_path = os.path.join(self.dataset_dir, 'landmarks', 'ref.json')
        with open(json_path) as json_file:
            json_data = json.load(json_file)
        return self._landmark_json_to_arr(json_data)

    def _landmark_json_to_arr(self, dct):
        all_points = []
        for loc in self.LANDMARK_LOCS:
            points = dct.get(loc)
            points = np.array(points).T  # output: (n_points, 2)
            all_points.extend(points)
        return np.array(all_points)

    def bbox_temp(self, im, bbox, agg_fn=np.mean):
        ylim, xlim = bbox
        return agg_fn(im[xlim[0]:xlim[1], ylim[0]:ylim[1]])

    def correct_nuc(self, im):
        # Calibrate temperature based on background
        bg = [self.bbox_temp(im, bbox) for bbox in self.BG_ROIS]
        bg = np.mean(bg)
        return im + (self.temp_env - bg), bg

    def align_ir_to_rgb(self, ir, rgb, ir_landmarks):
        w, h = RGB_IR_Data.CROP_WH

        # Warp IR image to align with RGB image based on camera homography.
        ir_warp = cv2.warpPerspective(ir, self.ir2rgb, (ir.shape[1], ir.shape[0]))

        # Crop IR and RGB image to the face based on detected landmarks
        rgb_landmarks = coords_ir_to_rgb(ir_landmarks)
        xmin, ymin = np.min(rgb_landmarks, axis=0)
        xmax, ymax = np.max(rgb_landmarks, axis=0)
        ymin = max(0, ymin - 25)    # extra offset for forehead
        ir_warp = ir_warp[ymin:ymax, xmin:xmax]
        rgb = rgb[ymin:ymax, xmin:xmax]

        # Resize all face images to the same shape: (90, 60)
        ir_warp = cv2.resize(ir_warp, (w, h))
        rgb = cv2.resize(rgb, (w, h))
        return ir_warp, rgb

    def save_ir(self):
        n = self.duration
        for i in range(n):
            ir = self.ir[:, :, i]
            raw = temp2raw(ir).astype(np.uint16)
            save_path = os.path.join(
                self.dataset_dir, 'ml_data',
                self.name, f'{self.session}_ir{i}.png',
            )
            cv2.imwrite(save_path, raw)
