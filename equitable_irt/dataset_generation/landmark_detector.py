import os

import cv2
import face_recognition
import imutils
import json
import numpy as np

from ..utils import bgr2rgb
from ..utils import load_im
from .face_utils import coords_rgb_to_ir


MIN_YCrCb = np.array([0, 135, 85], np.uint8)
MAX_YCrCb = np.array([235, 173, 127], np.uint8)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def _json_rgb_to_ir(data):
    out = {}
    for k in data:
        pts = data[k]
        if len(pts) == 0:
            continue
        out[k] = coords_rgb_to_ir(pts)
    return out


def _forehead(lms):
    brows = np.concatenate((lms['left_eyebrow'], lms['right_eyebrow']))
    x1 = brows[:, 0].min()
    x2 = brows[:, 0].max()
    y2 = brows[:, 1].min()
    if len(np.array(lms['forehead'])) == 0:
        y1 = y2 - 20
    else:
        y1 = np.array(lms['forehead'])[:, 1].min()
    xmin = x1 + (x2 - x1) // 3
    xmax = x2 - (x2 - x1) // 3
    ymin = y1 + (y2 - y1) // 2
    return [[xmin, ymin], [xmax, y2]]


def _left_eye(lms):
    x, y = lms['left_eye'][3]
    xmin, xmax = x - 4, x + 4
    ymin, ymax = y - 4, y + 4
    return [[xmin, ymin], [xmax, ymax]]


def _right_eye(lms):
    x, y = lms['right_eye'][0]
    xmin, xmax = x - 4, x + 4
    ymin, ymax = y - 4, y + 4
    return [[xmin, ymin], [xmax, ymax]]


def _left_cheek(lms):
    chin = np.array(lms['chin'])
    xmin = chin[:4, 0].max()
    xmax, ymax = lms['nose_tip'][0]
    ymin = lms['left_eye'][-2][1]
    return [[xmin, ymin], [xmax, ymax]]


def _right_cheek(lms):
    chin = np.array(lms['chin'])
    xmax = chin[-4:, 0].min()
    xmin, ymax = lms['nose_tip'][-1]
    ymin = lms['right_eye'][-2][1]
    return [[xmin, ymin], [xmax, ymax]]


def _left_eye_corner(lms):
    x, y = lms['left_eye'][0]
    return [[x - 3, y - 3], [x + 3, y + 3]]


def _right_eye_corner(lms):
    x, y = lms['right_eye'][3]
    return [[x - 3, y - 3], [x + 3, y + 3]]


def _left_nose(lms):
    x, y = lms['nose_tip'][0]
    return [[x - 3, y - 3], [x + 3, y + 3]]


def _right_nose(lms):
    x, y = lms['nose_tip'][-1]
    return [[x - 3, y - 3], [x + 3, y + 3]]


def _left_mouth(lms):
    x, y = lms['top_lip'][0]
    return [[x - 3, y - 3], [x + 3, y + 3]]


def _right_mouth(lms):
    x, y = lms['top_lip'][6]
    return [[x - 3, y - 3], [x + 3, y + 3]]


def _chin(lms):
    lip = np.array(lms['bottom_lip'])
    x, ymax = lms['chin'][8]
    ymin = lip[:, 1].max()
    return [[x - 3, ymin + 1], [x + 3, ymax - 1]]


def find_landmarks(image, num_forehead_landmarks=18):
    # Can process in batches but only matters it GPU is used
    lm_list = face_recognition.face_landmarks(image)
    if len(lm_list) == 0:
        return {}
    lm_list = lm_list[0]

    bounds = np.array([p for points in lm_list.values() for p in points])
    xmin, ymin, xmax, ymax = *np.min(bounds, axis=0), *np.max(bounds, axis=0)

    # Crop and mask image based on skin tone
    crop = image[:ymax, xmin:xmax]
    im_ycrcb = cv2.cvtColor(crop, cv2.COLOR_RGB2YCR_CB)
    mask = cv2.inRange(im_ycrcb, MIN_YCrCb, MAX_YCrCb)

    # Detect forehead contour using largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    pts = max_contour[max_contour[:, 0, 1] < ymin]
    pts = pts[::max(len(pts) // num_forehead_landmarks, 1)] + [xmin, 0]
    pts = pts[:, 0, :].tolist()
    forehead_landmarks = [tuple(x) for x in pts]
    lm_list['forehead'] = forehead_landmarks
    return lm_list


def find_regions(image, num_forehead_landmarks=18):
    lms = find_landmarks(image, num_forehead_landmarks)
    if len(lms) == 0:
        return lms
    data = {}
    rois = {}

    rois['forehead'] = _forehead(lms)
    rois['left_cheek'] = _left_cheek(lms)
    rois['right_cheek'] = _right_cheek(lms)
    rois['left_canthus'] = _left_eye(lms)
    rois['right_canthus'] = _right_eye(lms)
    rois['left_eye_corner'] = _left_eye_corner(lms)
    rois['right_eye_corner'] = _right_eye_corner(lms)
    rois['left_nose'] = _left_nose(lms)
    rois['right_nose'] = _right_nose(lms)
    rois['left_mouth'] = _left_mouth(lms)
    rois['right_mouth'] = _right_mouth(lms)
    rois['chin'] = _chin(lms)

    data['rois'] = _json_rgb_to_ir(rois)
    data['landmarks'] = _json_rgb_to_ir(lms)
    return data


def process_dir(im_dir, save_dir):
    num_forehead_landmarks = 10
    if save_dir is None:
        save_dir = im_dir
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    subdirs = os.listdir(im_dir)
    if '.DS_Store' in subdirs:
        subdirs.remove('.DS_Store')

    for dir in subdirs:
        path = os.path.join(im_dir, dir)
        data = {}
        for fname in os.listdir(path):
            if fname.endswith('.jpg') or fname.endswith('.jpeg'):
                im = load_im(os.path.join(path, fname), bgr2rgb)
                im = imutils.resize(im, height=120)
                data[fname] = find_regions(im, num_forehead_landmarks)
        with open(os.path.join(save_dir, f'{dir}.json'), 'w') as fp:
            json.dump(data, fp, sort_keys=False, indent=4, cls=NpEncoder)
    return data
