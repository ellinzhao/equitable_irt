import os

import cv2
import numpy as np

from ...utils import bgr2rgb
from ...utils import load_im
from ..face_utils import face_bbox


class RGB:

    WH = (213, 120)

    def __init__(self, dataset_dir, name, session, duration, landmarks, roi_pts, invalid):
        self.dataset_dir = dataset_dir
        self.name = name
        self.session = session
        self.duration = duration
        self.session_dir = os.path.join(dataset_dir, 'data', name, session)

        self.landmarks = landmarks
        self.roi_pts = roi_pts
        self.invalid = invalid
        self.data = self.load()

    def load(self):
        n = self.duration
        w, h = RGB.WH
        frames = np.zeros((h, w, 3, n))
        for i in range(n):
            path = os.path.join(self.session_dir, f'rgb{i}.jpg')
            im = load_im(path, transform=bgr2rgb)
            # Later data is full res is not resized so fix this
            frames[..., i] = cv2.resize(im, RGB.WH)
        return frames

    def random_im(self):
        # Pick an index where next 4 frames are valid.
        invalid_sec = []
        for i in range(0, self.duration, 1):
            j = np.arange(i, i + 1).astype(int)
            skip = np.any(self.invalid[j])
            invalid_sec += [skip]
        idx = np.arange(self.duration / 1)
        idx = idx[~np.array(invalid_sec)]
        i = np.random.choice(idx).astype(int) * 1

        roi = face_bbox(self.landmarks[i])
        (xmin, ymin), (xmax, ymax) = roi
        w, h = xmax - xmin, ymax - ymin
        offset = int((h - w) / 2)
        xmin -= offset
        xmax += offset
        square_roi = [(xmin, ymin), (xmax, ymax)]
        return i, self.data[..., i], square_roi
