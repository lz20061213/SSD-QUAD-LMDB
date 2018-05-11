# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import cv2
import numpy as np
import numpy.random as npr
npr.seed(3)
import sys
sys.path.append('../../')

from resize import Resizer
from expand import Expander
from distort import Distortor
from sample import Sampler

from config import cfg

if __name__ == '__main__':

    distorter = Distortor(**cfg.TRAIN.DISTORT_PARAM)
    expander = Expander(**cfg.TRAIN.EXPAND_PARAM)
    sampler = Sampler(cfg.TRAIN.SAMPLERS)
    resizer = Resizer(**cfg.TRAIN.RESIZE_PARAM)

    while True:
        im = cv2.imread('cat.jpg')
        gt_boxes = np.array([[0.33, 0.04, 0.71, 0.98]], dtype=np.float32)
        im = distorter.distort_image(im)
        im, gt_boxes = expander.expand_image(im, gt_boxes)
        im, gt_boxes = sampler.sample_image(im, gt_boxes)
        if len(gt_boxes) < 1: continue
        im = resizer.resize_image(im)
        for gt_box in gt_boxes:
            x1 = int(gt_box[0] * im.shape[1])
            y1 = int(gt_box[1] * im.shape[0])
            x2 = int(gt_box[2] * im.shape[1])
            y2 = int(gt_box[3] * im.shape[0])
            cv2.rectangle(im, (x1, y1), (x2, y2), (188, 119, 64), 2)
        cv2.imshow('Sample', im)
        cv2.waitKey(0)