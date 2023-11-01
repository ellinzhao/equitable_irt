import json

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from .sn_estimation import SN_MODEL


tform = np.array([
    [0.8745, 0.0234, -20.4140],
    [-0.0234, 0.8745, 10.8193],
    [0, 0, 1.0000],
])
rgb2ir = tform
ir2rgb = np.linalg.pinv(tform)


LANDMARK_LOCS = [
    'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip',
    'left_eye', 'right_eye', 'top_lip', 'bottom_lip',
]
BBOX_LOCS = [
    'forehead', 'left_cheek', 'right_cheek', 'left_canthus', 'right_canthus',
    'left_eye_corner', 'right_eye_corner', 'left_nose', 'right_nose',
    'left_mouth', 'right_mouth', 'chin',
]

MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
ORB = cv2.ORB_create(5000)


def coords_rgb_to_ir(pts):
    x = np.hstack((np.array(pts), np.ones((len(pts), 1))))
    y = np.dot(rgb2ir, x.T)
    return y[:2].astype(int)


def coords_ir_to_rgb(pts):
    x = np.hstack((np.array(pts), np.ones((len(pts), 1))))
    y = np.dot(ir2rgb, x.T)
    return y[:2].astype(int)


def surface_normals(im):
    tform = Compose([ToTensor(), Resize((256, 256))])
    im_t = torch.stack([tform(im[i].astype(np.uint8)) for i in range(len(im))], dim=0)
    normals = SN_MODEL(im_t)[0]
    normals = np.array(normals.data.permute(0, 2, 3, 1).cpu())
    normals = normals / np.expand_dims(np.sqrt(np.sum(normals**2, axis=-1)), -1)
    return normals


def is_symmetrical(lms):
    symmetrical = True
    if len(lms) == 0:
        return False
    new_lms = {}
    for feature in lms:
        pts = lms[feature]
        pts = coords_ir_to_rgb(np.array(pts).T)
        new_lms[feature] = pts

    x = []
    for feature in new_lms:
        x += list(new_lms[feature][0])
    symmetryLine = np.min(x) + (np.max(x) - np.min(x)) / 2

    for feature in new_lms:
        x = new_lms[feature][0]
        left = np.sum(x < symmetryLine)
        right = np.sum(x > symmetryLine)

        if feature == 'chin':
            if min(left, right) == 0:
                symmetrical = False
            elif np.abs(left - right) / min(left, right) > 0.4:
                symmetrical = False
        elif feature == 'left_eye' and right != 0:
            symmetrical = False
        elif feature == 'right_eye' and left != 0:
            symmetrical = False
    return symmetrical


def flatten_rois(rois, face_roi, keys):
    xmin, ymin, xmax, ymax = face_roi.astype(int)
    xoffset, yoffset = xmin, ymin
    curr_w = xmax - xmin
    arr = []
    for k in keys:
        (ymin, ymax), (xmin, xmax) = rois[k]
        pts = np.array([ymin - yoffset, ymax - yoffset, xmin - xoffset, xmax - xoffset])
        pts = pts / curr_w * 64  # This is hardcoded for now
        arr += list(pts.astype(int))
    return arr


def flatten_landmarks(dct):
    all_points = []
    for loc in LANDMARK_LOCS:
        points = dct.get(loc)
        points = np.array(points).T  # output: (n_points, 2)
        all_points.extend(points)
    return np.array(all_points)


def load_json(json_path, n):
    # Loads IR landmarks, IR ROIs and detect asymmetry
    # List: len=n_ims, each item is array shape (n, 2)
    invalid = []
    landmark_pts = []
    rois = []
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    for i in range(n):
        symmetric = True
        json_i = json_data.get(f'rgb{i}.jpg', {})
        roi_i = json_i.get('rois', {})
        if 'landmarks' not in json_i:
            print(f'Could not find landmarks: {i}')
            lms = [] if i == 0 else landmark_pts[-1]
            symmetric = False
        else:
            lms = flatten_landmarks(json_i.get('landmarks'))
        rois += [roi_i]
        landmark_pts += [lms]
        symmetric = symmetric and is_symmetrical(json_i.get('landmarks', {}))
        if not symmetric:
            invalid += [i]

    invalid = np.array(invalid, dtype=int)
    invalid_flag = np.zeros(n)
    invalid_flag[invalid] = 1
    return landmark_pts, rois, invalid_flag.astype(int)


def align_rgb(im1, im2):
    # im2 is the target image
    kp1, d1 = ORB.detectAndCompute(im1, None)
    kp2, d2 = ORB.detectAndCompute(im2, None)

    matches = list(MATCHER.match(d1, d2))
    matches.sort(key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 0.8)]
    n_matches = len(matches)

    p1 = np.zeros((n_matches, 2))
    p2 = np.zeros((n_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    if len(p2) < 4:
        return None

    M, _ = cv2.findHomography(p1, p2, cv2.RANSAC)
    return M


def face_bbox(lms, rois):
    (ymin, _), (_, _) = rois['forehead']  # ymin comes from forehead
    xmin, _ = np.min(lms, axis=0)
    xmax, ymax = np.max(lms, axis=0)
    ymin = max(0, ymin - 10)
    xmin = max(0, xmin - 10)
    ymax = min(120, ymax + 10)
    xmax = min(160, xmax + 10)
    h = ymax - ymin
    w = xmax - xmin
    offset = (h - w) // 2
    xmin -= offset
    xmax += offset
    return [(xmin, ymin), (xmax, ymax)]


def crop_resize(im, roi, m=64):
    (xmin, ymin), (xmax, ymax) = roi
    return cv2.resize(im[ymin:ymax, xmin:xmax], (m, m))


def frontal_pose(lms):
    # Accounts for roll and yaw, but not pitch changes

    # Check if the face is in frame
    xmin, ymin = lms.min(axis=0)
    xmax, ymax = lms.min(axis=0)
    vertical_check = ymin > 10 and ymax < 120  # forehead and chine not cut off
    horizontal_check = xmin > 0 and xmax < 160
    frame_check = vertical_check and horizontal_check

    # Check symmetry over the chin
    chin_lms = lms[:17]
    mid_chin = lms[8, 0]
    lchin = np.min(chin_lms[:, 0])
    rchin = np.max(chin_lms[:, 0])
    width = (rchin - lchin)
    true_mid = lchin + width / 2
    offcenter = np.abs(mid_chin - true_mid) / width

    # Check position of eyes
    leye_lms = lms[36:42]
    reye_lms = lms[42:48]
    lcenter = leye_lms.mean(axis=0)
    rcenter = reye_lms.mean(axis=0)
    slope = lcenter - rcenter
    angle = np.rad2deg(np.arctan(slope[1] / slope[0]))
    return frame_check and np.abs(angle) < 10 and offcenter < 0.15
