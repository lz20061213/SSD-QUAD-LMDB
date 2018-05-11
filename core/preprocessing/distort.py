# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import cv2
import PIL.Image
import PIL.ImageEnhance
import numpy as np
import numpy.random as npr
from config import cfg

class Distortor(object):
    def __init__(self, **params):
        self._brightness_prob = params.get('brightness_prob', 0.0)
        self._brightness_delta = 0.3
        self._contrast_prob = params.get('contrast_prob', 0.0)
        self._contrast_delta = 0.3
        self._saturation_prob = params.get('sauration_prob', 0.0)
        self._saturation_delta = 0.3

    def distort_image(self, im):
        im = PIL.Image.fromarray(im)
        if npr.uniform() < self._brightness_prob:
            delta_brightness = npr.uniform(-self._brightness_delta, self._brightness_delta) + 1.0
            im = PIL.ImageEnhance.Brightness(im)
            im = im.enhance(delta_brightness)
        if npr.uniform() < self._contrast_prob:
            delta_contrast = npr.uniform(-self._contrast_delta, self._contrast_delta) + 1.0
            im = PIL.ImageEnhance.Contrast(im)
            im = im.enhance(delta_contrast)
        if npr.uniform() < self._saturation_prob:
            delta_saturation = npr.uniform(-self._contrast_delta, self._contrast_delta) + 1.0
            im = PIL.ImageEnhance.Color(im)
            im = im.enhance(delta_saturation)
        im = np.array(im)
        return im


if __name__ == '__main__':

    distortor = Distortor(**cfg.TRAIN.DISTORT_PARAM)
    while True:
        im = cv2.imread('cat.jpg')
        im = distortor.distort_image(im)
        cv2.imshow('Distort', im)
        cv2.waitKey(0)