import cv2
import numpy as np
from PIL import Image
from scipy.constants import convert_temperature as conv_temp


DATETIME_TAG = 36867
bgr2rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def temp2raw(temp, units='F'):
    return 100 * conv_temp(temp, units, 'C') + 27315


def raw2temp(raw, units='F'):
    return conv_temp((raw - 27315) / 100, 'C', units)


def raw2viz(raw):
    # To make the raw images "previewable", convert to 8-bit
    bit = cv2.normalize(raw, None, 0, 65535, cv2.NORM_MINMAX)
    bit = np.right_shift(bit, 8)
    return bit.astype(np.uint8)


def load_im(path, transform=None):
    if not transform:
        transform = lambda raw: raw
    return transform(cv2.imread(path, -1))


def read_timestamp(path):
    im = Image.open(path)
    exif = im.getexif()
    return exif[DATETIME_TAG]


def adjust_gamma(im, gamma=1.0):
    # Uses lookup table for faster performance
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0)**invGamma) * 255 for i in range(256)
    ]).astype('uint8')
    return cv2.LUT(im, table)
