import json
import os

import cv2
import numpy as np
import scipy.signal as sps
from torch.autograd import Variable
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor

from utils import coords_ir_to_rgb
from utils import ir2rgb
from utils import load_im
from utils import raw2temp
from utils import temp2raw
from utils import is_symmetrical


class RGB_IR_Data:

    SESSION_DURATION = {
        'cool': 1 * 60 * 4,
        'base': 1 * 60 * 4,
    }

    RGB_WH = (213, 120)
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
                 correct_nuc=True, rois=['forehead'], sn_model=None):
        assert session in RGB_IR_Data.SESSION_DURATION.keys()
        self.duration = RGB_IR_Data.SESSION_DURATION.get(session)
        self.name = name
        self.session = session
        self.temp_env = temp_env
        self.dataset_dir = dataset_dir
        self.sn_model = sn_model

        # Load landmarks and detect if face is turned from landmarks
        self.landmarks, self.bboxes, self.face_turned_flag = self._load_landmarks()

        # Images are loaded as IR=120x160 and RGB=120x213
        # IR images are aligned to RGB and NUC corrected
        self.rois, self.ir, self.rgb = self._load_data(correct_nuc, rois)

        # Find peaks in IR data and flag frames near a NUC
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

    def _load_data(self, correct_nuc, rois):
        n = self.duration
        w, h = RGB_IR_Data.WH
        ir_images = np.zeros((h, w, n))

        w, h = RGB_IR_Data.RGB_WH
        rgb_images = np.zeros((h, w, 3, n))

        roi_temps = np.zeros((len(rois), n))
        bg_temps = np.zeros(n)

        for i in range(n):
            session_dir = os.path.join(self.dataset_dir, 'data', self.name, self.session)
            ir_path = os.path.join(session_dir, f'ir{i}.png')
            rgb_path = os.path.join(session_dir, f'rgb{i}.jpg')
            ir = load_im(ir_path, transform=raw2temp)
            rgb = load_im(rgb_path)

            # NUC correction and ROI calc must occur before image alignment
            if correct_nuc:
                ir, bg = self.correct_nuc(ir)
                bg_temps[i] = bg
            if rois:
                roi_temps[..., i] = self._load_rois(ir, self.bboxes[i], rois)

            # Align IR to RGB image based on camera homography
            ir = self.align_ir_to_rgb(ir)
            rgb = cv2.resize(rgb, (w, h))

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
        asymmetric_i = []
        landmark_arr = []
        bbox_arr = []
        with open(json_path) as json_file:
            json_data = json.load(json_file)

        for i in range(self.duration):
            symmetric = True
            json_i = json_data.get(f'rgb{i}.jpg', None)
            if ('landmarks' not in json_i) or (json_i is None):
                print(f'Could not find landmarks: {i}')
                lms = landmark_arr[-1]
                rois = bbox_arr[-1]
                symmetric = False
            else:
                lms = self._landmark_json_to_arr(json_i['landmarks'])
                rois = json_i['rois']

            # Check if subject is facing forward based on symmetry of landmarks
            symmetric = symmetric and is_symmetrical(json_i.get('landmarks', []))
            if not symmetric:
                asymmetric_i += [i]

            landmark_arr += [lms]
            bbox_arr += [rois]

        n = self.duration
        asymmetric_i = np.array(asymmetric_i, dtype=int)
        face_turned_flag = np.zeros(n)
        face_turned_flag[asymmetric_i] = 1
        return landmark_arr, bbox_arr, face_turned_flag

    def _landmark_json_to_arr(self, dct):
        all_points = []
        for loc in self.LANDMARK_LOCS:
            if 'brow' in loc or 'lip' in loc:
                continue
            points = dct.get(loc)
            points = np.array(points).T  # output: (n_points, 2)
            all_points.extend(points)
        return np.array(all_points)

    def _face_bbox(self, ir_lms):
        rgb_landmarks = coords_ir_to_rgb(ir_lms)
        xmin, ymin = np.min(rgb_landmarks, axis=0)
        xmax, ymax = np.max(rgb_landmarks, axis=0)
        ymin = max(0, ymin - 30)    # extra offset for forehead
        return [(xmin, ymin), (xmax, ymax)]
        # ir_warp = ir_warp[ymin:ymax, xmin:xmax]
        # rgb = rgb[ymin:ymax, xmin:xmax]

    def _surface_normal(self, im):
        tform = Compose([ToTensor()])
        im_t = tform(im.astype('float32')).unsqueeze(0)
        im_t = Variable(im_t)
        normals = self.sn_model(im_t)[0]
        normals = np.array(normals[0].data.permute(1, 2, 0).cpu())
        return normals

    def bbox_temp(self, im, bbox, agg_fn=np.mean):
        ylim, xlim = bbox
        return agg_fn(im[xlim[0]:xlim[1], ylim[0]:ylim[1]])

    def correct_nuc(self, im):
        # Calibrate temperature based on background
        bg = [self.bbox_temp(im, bbox) for bbox in self.BG_ROIS]
        bg = np.mean(bg)
        return im + (self.temp_env - bg), bg

    def align_ir_to_rgb(self, ir):
        # Warp IR image to align with RGB image based on camera homography.
        ir_warp = cv2.warpPerspective(ir, ir2rgb, (ir.shape[1], ir.shape[0]))
        return ir_warp

    def crop_data(self):
        n = self.duration
        rgb_proc = []
        ir_proc = []
        for i in range(n):
            ir = self.ir[..., i]
            rgb = self.rgb[..., i]
            lms = self.landmarks[i]
            (xmin, ymin), (xmax, ymax) = self._face_bbox(lms)
            h = ymax - ymin
            w = xmax - xmin
            offset = (h - w) // 2
            xmin -= offset
            xmax += offset
            ir_crop = cv2.resize(ir[ymin:ymax, xmin:xmax], (64, 64))
            rgb_crop = cv2.resize(rgb[ymin:ymax, xmin:xmax], (64, 64))

            ir_proc += [ir_crop]
            rgb_proc += [rgb_crop]
        self.rgb_proc = rgb_proc
        self.ir_proc = ir_proc

    def save_dataset(self):
        # Convert temp to raw 16-bit image
        n = self.duration
        for i in range(n):
            ir = self.ir_proc[i]
            rgb = self.rgb_proc[i]
            raw = temp2raw(ir).astype(np.uint16)
            ir_save_path = os.path.join(
                self.dataset_dir, 'ml_data', self.name, f'{self.session}_ir{i}.png',
            )
            cv2.imwrite(ir_save_path, raw)
            rgb_save_path = os.path.join(
                self.dataset_dir, 'ml_data', self.name, f'{self.session}_rgb{i}.png',
            )
            cv2.imwrite(rgb_save_path, rgb)
