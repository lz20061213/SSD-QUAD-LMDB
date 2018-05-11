# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# Modified by Zhuang Liu
# --------------------------------------------------------

import cv2
import numpy.random as npr
import numpy as np
import math
from config import cfg

class Expander(object):
    def __init__(self, **params):
        self._expand_prob = params.get('prob', 1.0)
        self._max_expand_ratio = params.get('max_expand_ratio', 1.0)
        assert self._max_expand_ratio >= 1.0

    def expand_image(self, im, gt_boxes=None):
        prob = npr.uniform()
        if prob > self._expand_prob : return im, gt_boxes
        ratio = npr.uniform(1.0, self._max_expand_ratio)
        if ratio == 1: return im, gt_boxes

        im_h = im.shape[0]
        im_w = im.shape[1]
        expand_h = int(im_h * ratio)
        expand_w = int(im_w * ratio)
        h_off = int(math.floor(npr.uniform(0.0, expand_h - im_h)))
        w_off = int(math.floor(npr.uniform(0.0, expand_w - im_w)))

        new_im = np.empty((expand_h, expand_w, 3), dtype=np.uint8)
        new_im.fill(127)
        new_im[h_off: h_off + im_h, w_off: w_off + im_w, :] = im

        # by lz.
        if gt_boxes is not None:
            ex_gt_boxes = gt_boxes.astype(gt_boxes.dtype, copy=True)
            ex_gt_boxes[:, 0:-1:2] = (gt_boxes[:, 0:-1:2] * im_w + w_off) / expand_w
            ex_gt_boxes[:, 1::2] = (gt_boxes[:, 1::2] * im_h + h_off) / expand_h
            return new_im, ex_gt_boxes
        return new_im, gt_boxes


if __name__ == '__main__':

    expander = Expander(**cfg.TRAIN.EXPAND_PARAM)
    while True:
        im = cv2.imread('cat.jpg')
        gt_boxes = np.array([[0.33, 0.04, 0.71, 0.98]], dtype=np.float32)
        im, gt_boxes = expander.expand_image(im, gt_boxes)
        x1 = int(gt_boxes[0][0] * im.shape[1])
        y1 = int(gt_boxes[0][1] * im.shape[0])
        x2 = int(gt_boxes[0][2] * im.shape[1])
        y2 = int(gt_boxes[0][3] * im.shape[0])
        cv2.rectangle(im, (x1, y1), (x2, y2), (188,119,64), 2)
        cv2.imshow('Expand', im)
        cv2.waitKey(0)




