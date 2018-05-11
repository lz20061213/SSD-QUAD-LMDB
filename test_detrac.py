# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.vm.caffe as caffe
from datasets.factory import get_imdb
from core.test import test_net
from config import cfg
import time, os

cfg.DATA_DIR = '/home/workspace/datasets/UA-DETRAC'
imdb_name = 'detrac_2017_test'
gpu_id = 1
prototxt = 'models/detrac/AirNet/deploy.prototxt'
caffemodel = 'checkpoints/airnet_final.caffemodel'
vis = False

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
