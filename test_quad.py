#!/usr/bin/env python
# encoding: utf-8

# --------------------------------------------------------
# file: test_quad.py
# Copyright(c) 2017-2020 SeetaTech
# Written by Zhuang Liu
# 上午10:09
# --------------------------------------------------------

import dragon.vm.caffe as caffe
from datasets.factory import get_imdb
from core.test import test_net
from config import cfg
import time, os

cfg.DATA_DIR = 'data/quad'
imdb_name = 'plane_2018_test'

gpu_id = 3
prototxt = 'models/quad/VGG16/300x300/test.prototxt'
caffemodel = 'output/plane_2018_trainval/QUAD_VGG16_iter_70000.caffemodel'
vis = True

if __name__ == '__main__':

    while not os.path.exists(caffemodel):
        print('Waiting for {} to exist...'.format(caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    net.name = os.path.splitext(os.path.basename(caffemodel))[0]
    imdb = get_imdb(imdb_name)

    test_net(net, imdb, thresh=cfg.TEST.THRESH, vis=vis)
